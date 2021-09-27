import argparse

import numpy as np
from torch.utils.data import DataLoader

from config.io import *
from data.utils import convert_array_to_dataset


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
    parser.add_argument('-consistency_loss', type=str, default='dice', help='consistency loss')
    parser.add_argument('-drop_rate', type=int, default=0.5, help='model drop rate')

    opt = parser.parse_args()
    return opt


def load_train_data(batch_size, num_workers):
    train_images_patches = np.load(TRAIN_IMAGES_PATCHES_PATH)
    train_masks_patches = np.load(TRAIN_MASKS_PATCHES_PATH)
    train_dataset = convert_array_to_dataset(train_images_patches, train_masks_patches)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True,
                                  num_workers=num_workers, pin_memory=True)
    return train_dataloader


def train(opt):
    train_dataloader = load_train_data(opt.batch_size, opt.num_workers)


if __name__ == '__main__':
    options = create_parser()
    train(options)
