from os import makedirs

import medicaltorch.filters as mt_filters
import medicaltorch.transforms as mt_transforms
import numpy as np
import torch
import torchvision as tv
from sklearn.feature_extraction.image import extract_patches_2d
from torch.utils.data import TensorDataset

from config.io import NPY_ROOT
from data.dataset import CC359


def get_dataset(img_root_dir, gt_root_dir, file_ids, slice_axis):
    dataset = CC359(img_root_dir=img_root_dir, gt_root_dir=gt_root_dir, slice_axis=slice_axis,
                    normalizer=min_max_normalization, slice_filter_fn=mt_filters.SliceFilter(), file_ids=file_ids)
    transform = tv.transforms.Compose([
        mt_transforms.ElasticTransform(alpha_range=(28.0, 30.0),
                                       sigma_range=(3.5, 4.0),
                                       p=0.3),
        mt_transforms.RandomAffine(degrees=4.6,
                                   scale=(0.98, 1.02),
                                   translate=(0.03, 0.03)),
        mt_transforms.RandomTensorChannelShift((-0.10, 0.10)),
        mt_transforms.ToTensor(),
        mt_transforms.NormalizeInstance(),
    ])

    dataset.set_transform(transform)
    return dataset


def patch_data(dataset, patch_size, max_patches):
    data_count = len(dataset)
    total_patches_count = data_count * max_patches
    images_patches = np.ndarray((total_patches_count, patch_size[0], patch_size[1]), dtype=np.float32)
    masks_patches = np.ndarray((total_patches_count, patch_size[0], patch_size[1]), dtype=np.uint8)

    random_value = np.random.randint(100)
    for i in range(1):
        data = dataset[i]
        image = data.get('input')[0]
        mask = data.get('gt')[0]
        image_patches = extract_patches_2d(np.array(image), patch_size, max_patches, random_state=random_value)
        mask_patches = extract_patches_2d(np.array(mask), patch_size, max_patches, random_state=random_value)
        images_patches[i * max_patches:max_patches * (i + 1)] = image_patches
        masks_patches[i * max_patches:max_patches * (i + 1)] = mask_patches

    return images_patches, masks_patches


def save_patches(images_patches, masks_patches, images_patches_path, masks_patches_path):
    makedirs(NPY_ROOT, exist_ok=True)
    np.save(images_patches_path, images_patches.astype(np.float32))
    np.save(masks_patches_path, masks_patches.astype(np.uint8))


def convert_array_to_dataset(x_arr, y_arr):
    x_tensor = torch.tensor(x_arr)
    y_tensor = torch.tensor(y_arr)
    dataset = TensorDataset(x_tensor, y_tensor)
    return dataset


def min_max_normalization(data):
    voxel_min = np.float32(np.min(data))
    voxel_max = np.float32(np.max(data))

    return (data - voxel_min) / (voxel_max - voxel_min)
