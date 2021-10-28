import argparse
import shutil
from os import listdir, makedirs

from sklearn.model_selection import train_test_split

from config.io import *
from config.param import Plane
from data.dataset import CC359
from data.utils import min_max_normalization


def split_data(data_dir, ratio):
    imgs_dir = os.path.join(data_dir, 'images')
    masks_dir = os.path.join(data_dir, 'masks')
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')

    file_ids = []
    mf = []
    for file_path in listdir(imgs_dir):
        if not file_path.startswith('.'):
            file_id = file_path.split('/')[-1].split('.')[0]
            file_ids.append(file_id)
            mf.append(file_id.split('_')[2])

    train_files, test_files = train_test_split(file_ids, test_size=ratio, random_state=42, shuffle=True, stratify=mf)

    copy_files(train_files, imgs_dir, os.path.join(train_dir, 'images'), IMAGE_FILE_TEMPLATE)
    copy_files(train_files, masks_dir, os.path.join(train_dir, 'masks'), MASK_FILE_TEMPLATE)
    copy_files(test_files, imgs_dir, os.path.join(test_dir, 'images'), IMAGE_FILE_TEMPLATE)
    copy_files(test_files, masks_dir, os.path.join(test_dir, 'masks'), MASK_FILE_TEMPLATE)

    return train_files, test_files


def copy_files(file_ids, source, dest, name_template):
    makedirs(dest, exist_ok=True)
    for file_id in file_ids:
        shutil.copy(os.path.join(source, name_template.format(file_id=file_id)), dest)


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_dir', type=str, default='imgs_train', help='data directory name')
    parser.add_argument('-num_epochs', type=int, default=5, help='number of epochs to train for')
    parser.add_argument('-test_ratio', type=int, default=0.2, help='')
    parser.add_argument('-normalize', type=bool, default=False, help='Do min max normalization on dataset')
    parser.add_argument('-plane', type=int, default=Plane.SAGITTAL.value, help='2D plane')

    opt = parser.parse_args()
    return opt


def preprocess(opt):
    data_dir = opt.data_dir
    slices_train_images_path = os.path.join(data_dir, 'slices/train/images')
    slices_train_masks_path = os.path.join(data_dir, 'slices/train/masks')
    slices_test_images_path = os.path.join(data_dir, 'slices/test/images')
    slices_test_masks_path = os.path.join(data_dir, 'slices/test/masks')

    train_files, test_files = split_data(data_dir, opt.test_ratio)
    train_set = CC359(os.path.join(data_dir, 'train/images'), os.path.join(data_dir, 'train/masks'), opt.plane,
                      train_files, normalizer=min_max_normalization)
    test_set = CC359(os.path.join(data_dir, 'test/images'), os.path.join(data_dir, 'test/masks'), opt.plane, test_files,
                     normalizer=min_max_normalization)
    train_set.save_slices(slices_train_images_path, slices_train_masks_path)
    test_set.save_slices(slices_test_images_path, slices_test_masks_path)


if __name__ == '__main__':
    options = create_parser()
    preprocess(options)
