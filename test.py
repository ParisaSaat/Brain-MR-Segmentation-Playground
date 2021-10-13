import albumentations as A
import torch
import time
from config.param import Plane
from os import listdir, makedirs
import argparse
from data.utils import get_dataset
from config.io import *


import albumentations.augmentations.functional as F
from albumentations.pytorch import ToTensorV2


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-imgs_dir', type=str, default='imgs_train', help='images directory name')
    parser.add_argument('-masks_dir', type=str, default='imgs_masks_train', help='mask directory name')
    opt = parser.parse_args()
    return opt


def main(opt):
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

    test_transform = A.Compose(
        [A.Resize(256, 256), ToTensorV2()]
    )
    val_dataset = get_dataset(SOURCE_TEST_IMAGES_PATH, SOURCE_TEST_MASKS_PATH, test_files, tag, val_transform)
    print('train dataset size:', len(train_dataset))


if __name__ == '__main__':
    options = create_parser()
    main(options)
