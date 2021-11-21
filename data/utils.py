from os import listdir
from os import makedirs

import numpy as np
import torch
from sklearn.feature_extraction.image import extract_patches_2d
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

from config.io import NPY_ROOT
from data.dataset import BrainMRI2D


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


def get_dataloader(image_dir, mask_dir, batch_size, transform, collate_fn=None, shuffle=True, drop_last=True,
                   pin_memory=True, num_workers=0, mean_teacher=False):
    image_files = listdir(image_dir)
    dataset = BrainMRI2D(image_dir, mask_dir, file_ids=image_files, transform=transform, mean_teacher=mean_teacher)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last,
                            num_workers=num_workers,
                            collate_fn=collate_fn,
                            pin_memory=pin_memory)
    return dataloader
