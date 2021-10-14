import argparse
import shutil
from os import listdir, makedirs

from sklearn.model_selection import train_test_split

from config.io import *
from config.param import Plane, TEST_RATIO
from data.dataset import CC359
from data.utils import min_max_normalization


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

    copy_files(train_files, imgs_dir, SOURCE_TRAIN_IMAGES_PATH, IMAGE_FILE_TEMPLATE)
    copy_files(train_files, masks_dir, SOURCE_TRAIN_MASKS_PATH, MASK_FILE_TEMPLATE)
    copy_files(test_files, imgs_dir, SOURCE_VAL_IMAGES_PATH, IMAGE_FILE_TEMPLATE)
    copy_files(test_files, masks_dir, SOURCE_VAL_MASKS_PATH, MASK_FILE_TEMPLATE)

    return train_files, test_files


def copy_files(file_ids, source, dest, name_template):
    makedirs(dest, exist_ok=True)
    for file_id in file_ids:
        shutil.copy(path.join(source, name_template.format(file_id=file_id)), dest)


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-imgs_dir', type=str, default='imgs_train', help='images directory name')
    parser.add_argument('-masks_dir', type=str, default='imgs_masks_train', help='mask directory name')
    parser.add_argument('-num_epochs', type=int, default=5, help='number of epochs to train for')
    parser.add_argument('-normalize', type=bool, default=False, help='Do min max normalization on dataset')
    parser.add_argument('-plane', type=int, default=Plane.SAGITTAL.value, help='2D plane')

    opt = parser.parse_args()
    return opt


def preprocess(opt):
    train_files, val_files = split_data(opt.imgs_dir, opt.masks_dir)
    train_set = CC359(SOURCE_TRAIN_IMAGES_PATH, SOURCE_TRAIN_MASKS_PATH, opt.plane, train_files,
                      normalizer=min_max_normalization)
    validation_set = CC359(SOURCE_VAL_IMAGES_PATH, SOURCE_VAL_MASKS_PATH, opt.plane, val_files,
                           normalizer=min_max_normalization)
    train_set.save_slices(SOURCE_SLICES_TRAIN_IMAGES_PATH, SOURCE_SLICES_TRAIN_MASKS_PATH)
    validation_set.save_slices(SOURCE_SLICES_VAL_IMAGES_PATH, SOURCE_SLICES_VAL_MASKS_PATH)


if __name__ == '__main__':
    options = create_parser()
    preprocess(options)
