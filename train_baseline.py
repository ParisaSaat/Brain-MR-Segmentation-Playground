import argparse
import time
from collections import defaultdict
from os import listdir

import albumentations as A
import medicaltorch.losses as mt_losses
import medicaltorch.metrics as mt_metrics
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import *
import random
from matplotlib import pyplot as plt

from config.io import *
from config.param import *
from data.dataset import BrainMRI2D
from models.baseline import Unet


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-batch_size', type=int, default=32, help='input batch size')
    parser.add_argument('-patch_height', type=int, default=128, help='patch size')
    parser.add_argument('-patch_width', type=int, default=128, help='patch size')
    parser.add_argument('-num_workers', type=int, default=16, help='number of workers')
    parser.add_argument('-num_epochs', type=int, default=10, help='number of epochs to train for')
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


def validation(model, loader, writer, metric_fns, epoch):
    val_loss = 0.0

    num_samples = 0
    num_steps = 0

    result_dict = defaultdict(float)

    for i, batch in enumerate(loader):
        image_data, mask_data = batch['image'], batch['mask']

        image_data_gpu = image_data.cuda()
        mask_data_gpu = mask_data.cuda()

        with torch.no_grad():
            model_out = model(image_data_gpu)
            val_class_loss = mt_losses.dice_loss(model_out, mask_data_gpu)
            val_loss += val_class_loss.item()

        masks = mask_data_gpu.cpu().numpy().astype(np.uint8)
        # print('masks.shape:', masks.shape)
        # masks = masks.squeeze(axis=1)

        predictions = model_out.cpu().numpy()
        predictions = predictions.squeeze(axis=1)
        predictions = predictions > 0.5

        for metric_fn in metric_fns:
            for prediction, mask in zip(predictions, masks):
                res = metric_fn(prediction, mask)
                dict_key = 'val_{}'.format(metric_fn.__name__)
                result_dict[dict_key] += res
                chance = random.uniform(0, 1)
                if chance < PLOTTING_RATE:
                    plt.imshow(prediction, cmap='gray')
                    plt.savefig('val_examples/{}_{}.png'.format(epoch, chance))
                    plt.imshow(mask, cmap='gray')
                    plt.savefig('val_examples/{}_{}_mask.png'.format(epoch, chance))

        num_samples += len(predictions)
        num_steps += 1

    val_loss_avg = val_loss / num_steps

    for key, val in result_dict.items():
        result_dict[key] = val / num_samples

    writer.add_scalars('losses', {'loss': val_loss_avg}, epoch)
    writer.add_scalars('metrics', result_dict, epoch)


def get_dataloader(image_dir, mask_dir, batch_size, transform):
    image_files = listdir(image_dir)
    dataset = BrainMRI2D(image_dir, mask_dir, file_ids=image_files, transform=transform)
    dataloader = DataLoader(dataset, batch_size)
    return dataloader


def train(opt):
    if torch.cuda.is_available():
        print('cuda is available')
        torch.cuda.set_device("cuda:0")

    patch_size = (opt.patch_height, opt.patch_width)

    train_transform = A.Compose(
        [
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=0.5),
            A.RandomCrop(height=patch_size[0], width=patch_size[1]),
            A.RandomBrightnessContrast(p=0.5),
            ToTensorV2(),
        ]
    )

    val_transform = A.Compose(
        [
            A.CenterCrop(height=patch_size[0], width=patch_size[1]),
            ToTensorV2(),
        ]
    )

    train_dataloader = get_dataloader(SOURCE_SLICES_TRAIN_IMAGES_PATH, SOURCE_SLICES_TRAIN_MASKS_PATH, opt.batch_size,
                                      train_transform)
    validation_dataloader = get_dataloader(SOURCE_SLICES_VAL_IMAGES_PATH, SOURCE_SLICES_VAL_IMAGES_PATH, opt.batch_size,
                                           val_transform)

    model = Unet(drop_rate=opt.drop_rate)
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
            train_image, train_mask = train_batch['image'], train_batch['mask']
            train_image = train_image.cuda()
            train_mask = train_mask.cuda()
            prediction = model(train_image)
            loss = mt_losses.dice_loss(prediction, train_mask)
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            loss_total += loss.item()
            #
            num_steps += 1

        loss_avg = loss_total / num_steps

        tqdm.write("Steps p/ Epoch: {}".format(num_steps))
        tqdm.write("Class Loss: {:.6f}".format(loss_avg))

        writer.add_scalars('losses', {'loss': loss_avg}, epoch)

        model.eval()

        metric_fns = [mt_metrics.dice_score, mt_metrics.jaccard_score, mt_metrics.hausdorff_score,
                      mt_metrics.precision_score, mt_metrics.recall_score,
                      mt_metrics.specificity_score, mt_metrics.intersection_over_union,
                      mt_metrics.accuracy_score]

        validation(model, validation_dataloader, writer, metric_fns, epoch)
        torch.save(model.state_dict(), MODEL_PATH.format(model_name='base_line'))

        end_time = time.time()
        total_time = end_time - start_time
        tqdm.write("Epoch {} took {:.2f} seconds.".format(epoch, total_time))


if __name__ == '__main__':
    options = create_parser()
    train(options)
