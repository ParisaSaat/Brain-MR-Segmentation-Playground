import argparse
import time
from collections import defaultdict

import medicaltorch.losses as mt_losses
import medicaltorch.metrics as mt_metrics
import numpy as np
import torch
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import *

from config.io import *
from config.param import LAMBDA
from data.utils import convert_array_to_dataset
from models.baseline import Unet


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-batch_size', type=int, default=16, help='input batch size')
    parser.add_argument('-num_workers', type=int, default=16, help='number of workers')
    parser.add_argument('-num_epochs', type=int, default=1, help='number of epochs to train for')
    parser.add_argument('-experiment_name', type=str, default='', help='experiment name')
    parser.add_argument('-initial_lr', type=float, default=1e-3, help='learning rate of the optimizer')
    parser.add_argument('-initial_lr_rampup', type=float, default=50, help='initial learning rate rampup')
    parser.add_argument('-decay', type=float, default=0.995, help='learning rate of the optimizer')
    parser.add_argument('-validation_split', type=float, default=0.1, help='validation split for training')
    parser.add_argument('-patience', type=int, default=10, help='early stopping patience')
    parser.add_argument('-drop_rate', type=float, default=0.5, help='model drop rate')
    parser.add_argument('-write_images_interval', type=int, default=20, help='write sample images in every interval')
    parser.add_argument('-write_images', type=bool, default=True, help='write sample images')

    opt = parser.parse_args()
    return opt


def cosine_rampdown(current, rampdown_length):
    """Cosine rampdown from https://arxiv.org/abs/1608.03983"""
    assert 0 <= current <= rampdown_length
    return float(.5 * (np.cos(np.pi * current / rampdown_length) + 1))


def cosine_lr(current_epoch, num_epochs, initial_lr):
    return initial_lr * cosine_rampdown(current_epoch, num_epochs)


def sigmoid_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def get_current_consistency_weight(weight, epoch, rampup):
    """Consistency ramp-up from https://arxiv.org/abs/1610.02242"""
    return weight * sigmoid_rampup(epoch, rampup)


def load_data_patches(img_path, msk_path, batch_size, num_workers):
    images_patches = np.load(img_path)
    masks_patches = np.load(msk_path)
    dataset = convert_array_to_dataset(images_patches, masks_patches)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers,
                            pin_memory=True)
    return dataloader


def validation(model, loader, writer, metric_fns, epoch, prefix):
    val_loss = 0.0

    num_samples = 0
    num_steps = 0

    result_dict = defaultdict(float)

    for i, batch in enumerate(loader):
        image_data, mask_data = batch["input"], batch["gt"]

        image_data_gpu = image_data.cuda()
        mask_data_gpu = image_data.cuda()

        with torch.no_grad():
            model_out = model(image_data_gpu)
            val_class_loss = mt_losses.dice_loss(model_out, mask_data_gpu)
            val_loss += val_class_loss.item()

        masks = mask_data_gpu.cpu().numpy().astype(np.uint8)
        masks = masks.squeeze(axis=1)

        preds = model_out.cpu().numpy()
        preds = preds.squeeze(axis=1)

        for metric_fn in metric_fns:
            for prediction, mask in zip(preds, masks):
                res = metric_fn(prediction, mask)
                dict_key = 'val_{}'.format(metric_fn.__name__)
                result_dict[dict_key] += res

        num_samples += len(preds)
        num_steps += 1

    val_loss_avg = val_loss / num_steps

    for key, val in result_dict.items():
        result_dict[key] = val / num_samples

    writer.add_scalars(prefix + '_losses', {prefix + '_loss': val_loss_avg}, epoch)
    writer.add_scalars(prefix + '_metrics', result_dict, epoch)


def train(opt):
    if torch.cuda.is_available():
        torch.cuda.set_device("cuda:0")

    train_dataloader = load_data_patches(TRAIN_IMAGES_PATCHES_PATH, TRAIN_MASKS_PATCHES_PATH, opt.batch_size,
                                         opt.num_workers)
    valid_dataloader = load_data_patches(TEST_IMAGES_PATCHES_PATH, TEST_MASKS_PATCHES_PATH, opt.batch_size,
                                         opt.num_workers)

    model = Unet(drop_rate=opt.drop_rate, bn_momentum=opt.bn_momentum)
    model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), weight_decay=LAMBDA, lr=opt.initial_lr)
    initial_lr = opt.initial_lr
    num_epochs = opt.num_epochs

    writer = SummaryWriter(log_dir="log_{}".format(opt.experiment_name))

    for epoch in tqdm(range(1, num_epochs + 1), desc="Epochs"):
        start_time = time.time()

        initial_lr_rampup = opt.initial_lr_rampup
        if epoch <= initial_lr_rampup:
            lr = initial_lr * sigmoid_rampup(epoch, initial_lr_rampup)
        else:
            lr = cosine_lr(epoch - initial_lr_rampup, num_epochs - initial_lr_rampup, initial_lr)

        writer.add_scalar('learning_rate', lr, epoch)

        for param_group in optimizer.param_groups:
            tqdm.write("Learning Rate: {:.6f}".format(lr))
            param_group['lr'] = lr

        model.train()

        loss_total = 0.0

        num_steps = 0

        for i, train_batch in enumerate(train_dataloader):
            train_image, train_mask = train_batch["input"], train_batch["gt"]
            train_image = train_image.cuda()
            train_mask = train_mask.cuda()
            prediction = model(train_image)
            loss = mt_losses.dice_loss(prediction, train_mask)
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            loss_total += loss.item()

            num_steps += 1

        npy_prediction = prediction.detach().cpu().numpy()
        writer.add_histogram("Prediction Hist", npy_prediction, epoch)

        loss_avg = loss_total / num_steps

        tqdm.write("Steps p/ Epoch: {}".format(num_steps))
        tqdm.write("Class Loss: {:.6f}".format(loss_avg))

        if opt.write_images and epoch % opt.write_images_interval == 0:
            try:
                plot_img = vutils.make_grid(prediction, normalize=True, scale_each=True)
                writer.add_image('Train Source Prediction', plot_img, epoch)

                plot_img = vutils.make_grid(train_image, normalize=True, scale_each=True)
                writer.add_image('Train Source Input', plot_img, epoch)

                plot_img = vutils.make_grid(train_mask, normalize=True, scale_each=True)
                writer.add_image('Train Source Ground Truth', plot_img, epoch)
            except:
                tqdm.write("*** Error writing images ***")

        writer.add_scalars('losses', {'loss': loss_avg}, epoch)

        model.eval()

        metric_fns = [mt_metrics.dice_score, mt_metrics.jaccard_score, mt_metrics.hausdorff_score,
                      mt_metrics.precision_score, mt_metrics.recall_score,
                      mt_metrics.specificity_score, mt_metrics.intersection_over_union,
                      mt_metrics.accuracy_score]

        validation(model, valid_dataloader, writer, metric_fns, epoch, opt)

        end_time = time.time()
        total_time = end_time - start_time
        tqdm.write("Epoch {} took {:.2f} seconds.".format(epoch, total_time))


if __name__ == '__main__':
    options = create_parser()
    train(options)
