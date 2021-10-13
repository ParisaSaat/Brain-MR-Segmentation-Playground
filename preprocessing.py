import argparse
import shutil
import time
from os import listdir, makedirs

import torch
from sklearn.model_selection import train_test_split

from config.io import *
from config.param import Plane, TEST_RATIO
from data.utils import get_dataset, patch_data, save_patches, min_max_normalization
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import os

import albumentations as A
import albumentations.augmentations.functional as F
from albumentations.pytorch import ToTensorV2


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-patch_size_row', type=int, default='128', help='patch width')
    parser.add_argument('-patch_size_col', type=int, default='128', help='patch height')
    parser.add_argument('-max_patches', type=int, default=3, help='number of patches per slice')
    parser.add_argument('-imgs_dir', type=str, default='imgs_train', help='images directory name')
    parser.add_argument('-masks_dir', type=str, default='imgs_masks_train', help='mask directory name')
    parser.add_argument('-num_epochs', type=int, default=5, help='number of epochs to train for')
    parser.add_argument('-normalize', type=bool, default=False, help='Do min max normalization on dataset')

    opt = parser.parse_args()
    return opt


def augment_data(dataset, transforms):
    dataset.set_transform(transforms)


def preprocess(opt):
    patch_size = opt.patch_size_row, opt.patch_size_col

    if torch.cuda.is_available():
        torch.cuda.set_device("cuda:0")

    start_time = time.time()
    tag = Plane.SAGITTAL.value
    file_ids = []
    vendors = []
    for file_path in listdir(opt.imgs_dir):
        if not file_path.startswith('.'):
            file_id = file_path.split('/')[-1].split('.')[0]
            file_ids.append(file_id)
            vendors.append(file_id.split('_')[1])

    train_files, test_files = train_test_split(file_ids, test_size=TEST_RATIO, random_state=42, shuffle=True,
                                               stratify=vendors)

    makedirs(SOURCE_TRAIN_IMAGES_PATH, exist_ok=True)
    makedirs(SOURCE_TRAIN_MASKS_PATH, exist_ok=True)
    makedirs(SOURCE_TEST_IMAGES_PATH, exist_ok=True)
    makedirs(SOURCE_TEST_MASKS_PATH, exist_ok=True)

    for file_id in train_files:
        shutil.copy(path.join(opt.imgs_dir, IMAGE_FILE_TEMPLATE.format(file_id=file_id)), SOURCE_TRAIN_IMAGES_PATH)
        shutil.copy(path.join(opt.masks_dir, MASK_FILE_TEMPLATE.format(file_id=file_id)), SOURCE_TRAIN_MASKS_PATH)
    for file_id in test_files:
        shutil.copy(path.join(opt.imgs_dir, IMAGE_FILE_TEMPLATE.format(file_id=file_id)), SOURCE_TEST_IMAGES_PATH)
        shutil.copy(path.join(opt.masks_dir, MASK_FILE_TEMPLATE.format(file_id=file_id)), SOURCE_TEST_MASKS_PATH)

    train_dataset = get_dataset(SOURCE_TRAIN_IMAGES_PATH, SOURCE_TRAIN_MASKS_PATH, train_files, tag)
    test_dataset = get_dataset(SOURCE_TEST_IMAGES_PATH, SOURCE_TEST_MASKS_PATH, test_files, tag)
    print('train dataset size:', len(train_dataset))
    print('val dataset size:', len(val_dataset))
    figure, ax = plt.subplots(nrows=opt.num_epochs, ncols=4, figsize=(10, 24))
    for i in range(opt.num_epochs):
        # train_images_patches, train_masks_patches = patch_data(train_dataset, patch_size, opt.max_patches)
        # val_images_patches, test_masks_patches = patch_data(val_dataset, patch_size, opt.max_patches)
        #
        # save_patches(train_images_patches, train_masks_patches, TRAIN_IMAGES_PATCHES_PATH, TRAIN_MASKS_PATCHES_PATH)
        # save_patches(val_images_patches, test_masks_patches, TEST_IMAGES_PATCHES_PATH, TEST_MASKS_PATCHES_PATH)
        image = train_dataset[50].get('input')
        mask = train_dataset[50].get('gt')
        asl = train_dataset[50].get('asl')
        print('input_img:', type(asl), asl.shape)
        print(asl)
        asl_mask = train_dataset[50].get('asl_mask')
        ax[i, 0].imshow(image[0], cmap='gray')
        ax[i, 1].imshow(mask, interpolation="nearest", cmap='gray')
        ax[i, 2].imshow(asl, cmap='gray')
        ax[i, 3].imshow(asl_mask, interpolation="nearest", cmap='gray')
        ax[i, 0].set_title("Augmented image")
        ax[i, 1].set_title("Augmented mask")
        ax[i, 2].set_title("asl image")
        ax[i, 3].set_title("asl mask")
        ax[i, 0].set_axis_off()
        ax[i, 1].set_axis_off()
        ax[i, 2].set_axis_off()
        ax[i, 3].set_axis_off()
        plt.tight_layout()
        plt.show()

    print("--- %s seconds ---" % (time.time() - start_time))


def offline_normalization(images_path):
    makedirs(NORMALIZED_IMAGES_PATH, exist_ok=True)
    for file_id in listdir(images_path):
        if not file_id.startswith('.'):
            image = nib.load(os.path.join(images_path, file_id))
            image_data = image.get_fdata(dtype=np.float32)
            normalized_image = min_max_normalization(image_data)
            niftii_img = nib.Nifti1Image(normalized_image, np.eye(4))
            nib.save(niftii_img, os.path.join(NORMALIZED_IMAGES_PATH,
                                              NORMALIZED_IMAGES_TEMPLATE.format(file_id=file_id.split('.')[0])))


if __name__ == '__main__':
    options = create_parser()
    if options.normalize:
        offline_normalization(options.imgs_dir)
    else:
        preprocess(options)
