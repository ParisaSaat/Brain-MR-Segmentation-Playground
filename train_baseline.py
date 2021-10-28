import argparse
import os.path
import random
import time
from collections import defaultdict
from os import makedirs

import albumentations as A
import medicaltorch.losses as mt_losses
import medicaltorch.metrics as mt_metrics
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from matplotlib import pyplot as plt
from tensorboardX import SummaryWriter
from tqdm import *

from config.io import *
from config.param import *
from data.utils import get_dataloader
from models.baseline import Unet
from models.utils import EarlyStopping, scheduler


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-batch_size', type=int, default=32, help='input batch size')
    parser.add_argument('-patch_height', type=int, default=128, help='patch size')
    parser.add_argument('-patch_width', type=int, default=128, help='patch size')
    parser.add_argument('-num_workers', type=int, default=16, help='number of workers')
    parser.add_argument('-num_epochs', type=int, default=30, help='number of epochs to train for')
    parser.add_argument('-experiment_name', type=str, default='baseline', help='experiment name')
    parser.add_argument('-initial_lr', type=float, default=5e-4, help='learning rate of the optimizer')
    parser.add_argument('-initial_lr_rampup', type=float, default=50, help='initial learning rate rampup')
    parser.add_argument('-decay', type=float, default=0.995, help='learning rate of the optimizer')
    parser.add_argument('-validation_split', type=float, default=0.1, help='validation split for training')
    parser.add_argument('-patience', type=int, default=50, help='early stopping patience')
    parser.add_argument('-drop_rate', type=float, default=0.5, help='model drop rate')
    parser.add_argument('-write_images_interval', type=int, default=20, help='write sample images in every interval')
    parser.add_argument('-write_images', type=bool, default=True, help='write sample images')
    parser.add_argument('-train_dir', type=str, default='Directory of training data', help='train data directory')
    parser.add_argument('-val_dir', type=str, default='Directory of validation data', help='validation data directory')

    opt = parser.parse_args()
    return opt


def validation(model, loader, writer, metric_fns, epoch, val_samples_dir):
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
            dice_loss = mt_losses.dice_loss(model_out, mask_data_gpu)
            val_loss += dice_loss.item()

        masks = mask_data_gpu.cpu().numpy().astype(np.uint8)
        imgs = image_data_gpu.cpu().numpy().astype(np.uint8)
        predictions = model_out.cpu().numpy()
        predictions = predictions.squeeze(axis=1)

        for metric_fn in metric_fns:
            for prediction, mask, img in zip(predictions, masks, imgs):
                res = metric_fn(prediction, mask)
                dict_key = 'val_{}'.format(metric_fn.__name__)
                result_dict[dict_key] += res
                chance = random.uniform(0, 1)
                if chance < PLOTTING_RATE:
                    plt.imshow(prediction > 0.5, cmap='gray')
                    plt.savefig(val_samples_dir + '/{}_{}_pred.png'.format(epoch, chance))
                    plt.imshow(mask > 0.5, cmap='gray')
                    plt.savefig(val_samples_dir + '/{}_{}_mask.png'.format(epoch, chance))

        num_samples += len(predictions)
        num_steps += 1

    val_loss_avg = val_loss / num_steps

    for key, val in result_dict.items():
        result_dict[key] = val / num_samples

    writer.add_scalars('losses', {'loss': val_loss_avg}, epoch)
    writer.add_scalars('metrics', result_dict, epoch)
    return val_loss_avg


def train(opt):
    experiment_name = opt.experiment_name
    train_dir = opt.train_dir
    val_dir = opt.val_dir
    val_samples_dir = 'val_samples_{}'.format(experiment_name)
    makedirs(val_samples_dir)
    if torch.cuda.is_available():
        print('cuda is available')
        torch.cuda.set_device("cuda:0")

    lr = opt.initial_lr

    patch_size = (opt.patch_height, opt.patch_width)

    train_transform = A.Compose(
        [
            A.ShiftScaleRotate(),
            A.RandomCrop(height=patch_size[0], width=patch_size[1]),
            ToTensorV2(),
        ]
    )

    val_transform = A.Compose(
        [
            A.CenterCrop(height=patch_size[0], width=patch_size[1]),
            ToTensorV2(),
        ]
    )

    train_dataloader = get_dataloader(os.path.join(train_dir, 'images'), os.path.join(train_dir, 'masks'),
                                      opt.batch_size, train_transform)
    validation_dataloader = get_dataloader(os.path.join(val_dir, 'images'), os.path.join(val_dir, 'masks'),
                                           opt.batch_size, val_transform)

    model = Unet(drop_rate=opt.drop_rate)
    model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), weight_decay=LAMBDA, lr=opt.initial_lr)
    num_epochs = opt.num_epochs
    early_stopping = EarlyStopping(opt.patience, 0.002)

    writer = SummaryWriter(log_dir="log_{}".format(opt.experiment_name))

    for epoch in tqdm(range(1, num_epochs + 1), desc="Epochs"):
        start_time = time.time()

        lr = scheduler(epoch, lr)

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
            num_steps += 1

        loss_avg = loss_total / num_steps

        tqdm.write("Steps p/ Epoch: {}".format(num_steps))
        tqdm.write("Train Loss: {:.6f}".format(loss_avg))

        writer.add_scalars('losses', {'loss': loss_avg}, epoch)

        model.eval()

        metric_fns = [mt_metrics.dice_score, mt_metrics.jaccard_score, mt_metrics.hausdorff_score,
                      mt_metrics.precision_score, mt_metrics.recall_score,
                      mt_metrics.specificity_score, mt_metrics.intersection_over_union,
                      mt_metrics.accuracy_score]

        val_loss = validation(model, validation_dataloader, writer, metric_fns, epoch, val_samples_dir)
        tqdm.write("Validation Loss: {:.6f}".format(val_loss))
        early_stop = early_stopping(val_loss)
        torch.save(model, MODEL_PATH.format(model_name='{}'.format(experiment_name)))

        end_time = time.time()
        total_time = end_time - start_time
        tqdm.write("Epoch {} took {:.2f} seconds.".format(epoch, total_time))
        if early_stop:
            break


if __name__ == '__main__':
    options = create_parser()
    train(options)
