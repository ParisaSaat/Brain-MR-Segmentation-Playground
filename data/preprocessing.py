import argparse
import shutil
from os import listdir, makedirs

from sklearn.model_selection import train_test_split

from config.io import *
from config.param import Plane
from data.dataset import CC359


def split_data(data_dir, ratio, image_path, mask_path, mask_type):
    imgs_dir = os.path.join(data_dir, image_path)
    masks_dir = os.path.join(data_dir, mask_path)
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    test_dir = os.path.join(data_dir, 'test')

    file_ids = []
    for file_path in listdir(imgs_dir):
        if not file_path.startswith('.'):
            file_id = file_path.split('/')[-1].split('.')[0]
            file_ids.append(file_id)

    train_files, rem_files = train_test_split(file_ids, test_size=ratio, random_state=42, shuffle=True)
    val_files, test_files = train_test_split(rem_files, test_size=0.5, random_state=42, shuffle=True)
    mask_file_template = '{file_id}_staple.nii.gz' if mask_type == 'staple' else '{file_id}_pveseg.nii.gz'
    copy_files(train_files, imgs_dir, os.path.join(train_dir, image_path), IMAGE_FILE_TEMPLATE)
    copy_files(train_files, masks_dir, os.path.join(train_dir, mask_path), mask_file_template)
    copy_files(val_files, imgs_dir, os.path.join(val_dir, image_path), IMAGE_FILE_TEMPLATE)
    copy_files(val_files, masks_dir, os.path.join(val_dir, mask_path), mask_file_template)
    copy_files(test_files, imgs_dir, os.path.join(test_dir, image_path), IMAGE_FILE_TEMPLATE)
    copy_files(test_files, masks_dir, os.path.join(test_dir, mask_path), mask_file_template)
    with open(os.path.join(data_dir, 'train_files.txt'), 'w') as f:
        for file in train_files:
            f.write("%s\n" % file)
    with open(os.path.join(data_dir, 'val_files.txt'), 'w') as f:
        for file in val_files:
            f.write("%s\n" % file)
    with open(os.path.join(data_dir, 'test_files.txt'), 'w') as f:
        for file in test_files:
            f.write("%s\n" % file)

    return train_files, val_files, test_files


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
    parser.add_argument('-problem', type=str, default='skull-stripping', help='segmentation problem')

    opt = parser.parse_args()
    return opt


def preprocess(opt):
    data_dir = opt.data_dir
    mask_type = 'pveseg' if opt.problem == 'wgc' else 'staple'
    img_pth = 'images_wgc' if opt.problem == 'wgc' else 'images'
    msk_pth = 'masks_wgc' if opt.problem == 'wgc' else 'masks'
    slices_train_images_path = os.path.join(data_dir, 'slices/train', img_pth)
    slices_train_masks_path = os.path.join(data_dir, 'slices/train', msk_pth)
    slices_val_images_path = os.path.join(data_dir, 'slices/val', img_pth)
    slices_val_masks_path = os.path.join(data_dir, 'slices/val', msk_pth)
    slices_test_images_path = os.path.join(data_dir, 'slices/test', img_pth)
    slices_test_masks_path = os.path.join(data_dir, 'slices/test', msk_pth)

    train_files, val_files, test_files = split_data(data_dir, opt.test_ratio, img_pth, msk_pth, mask_type)
    train_set = CC359(os.path.join(data_dir, 'train', img_pth), os.path.join(data_dir, 'train', msk_pth), opt.plane,
                      train_files, normalizer=None, mask_type=mask_type)
    val_set = CC359(os.path.join(data_dir, 'val', img_pth), os.path.join(data_dir, 'val', msk_pth), opt.plane,
                    val_files, normalizer=None, mask_type=mask_type)
    test_set = CC359(os.path.join(data_dir, 'test', img_pth), os.path.join(data_dir, 'test', msk_pth), opt.plane,
                     test_files, normalizer=None, mask_type=mask_type)
    train_set.save_slices(slices_train_images_path, slices_train_masks_path)
    val_set.save_slices(slices_val_images_path, slices_val_masks_path)
    test_set.save_slices(slices_test_images_path, slices_test_masks_path)


if __name__ == '__main__':
    options = create_parser()
    preprocess(options)
