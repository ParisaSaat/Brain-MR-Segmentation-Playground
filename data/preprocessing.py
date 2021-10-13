import argparse

from config.param import Plane, TEST_RATIO
from data.dataset import CC359
from data.utils import min_max_normalization, get_dataset
import medicaltorch.filters as mt_filters
from os import listdir, makedirs
from sklearn.model_selection import train_test_split
from config.io import *
import shutil


def split_data(imgs_dir, masks_dir):
    file_ids = []
    vendors = []
    for file_path in listdir(imgs_dir):
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
        shutil.copy(path.join(imgs_dir, IMAGE_FILE_TEMPLATE.format(file_id=file_id)), SOURCE_TRAIN_IMAGES_PATH)
        shutil.copy(path.join(masks_dir, MASK_FILE_TEMPLATE.format(file_id=file_id)), SOURCE_TRAIN_MASKS_PATH)
    for file_id in test_files:
        shutil.copy(path.join(imgs_dir, IMAGE_FILE_TEMPLATE.format(file_id=file_id)), SOURCE_TEST_IMAGES_PATH)
        shutil.copy(path.join(masks_dir, MASK_FILE_TEMPLATE.format(file_id=file_id)), SOURCE_TEST_MASKS_PATH)
    train_dataset = get_dataset(SOURCE_TRAIN_IMAGES_PATH, SOURCE_TRAIN_MASKS_PATH, train_files, tag)
    val_dataset = get_dataset(SOURCE_VAL_IMAGES_PATH, SOURCE_VAL_MASKS_PATH, test_files, tag)
    print('train dataset size:', len(train_dataset))
    print('val dataset size:', len(val_dataset))
    return train_dataset, val_dataset


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-imgs_dir', type=str, default='imgs_train', help='images directory name')
    parser.add_argument('-masks_dir', type=str, default='imgs_masks_train', help='mask directory name')
    parser.add_argument('-num_epochs', type=int, default=5, help='number of epochs to train for')
    parser.add_argument('-normalize', type=bool, default=False, help='Do min max normalization on dataset')

    opt = parser.parse_args()
    return opt


def preprocess(opt):
    train_dataset, validation_dataset = split_data(opt.imgs_dir, opt.masks_dir)
    train_dataset.save(SOURCE_TRAIN_IMAGES_PATH, SOURCE_TRAIN_MASKS_PATH)
    validation_dataset.save(SOURCE_VAL_IMAGES_PATH, SOURCE_VAL_MASKS_PATH)




if __name__ == '__main__':
    options = create_parser()
    preprocess(options)
