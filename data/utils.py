from os import listdir

import medicaltorch.datasets as mt_datasets
import medicaltorch.filters as mt_filters
import medicaltorch.transforms as mt_transforms
import numpy as np
import torchvision as tv
from torch.utils.data import DataLoader

from data.dataset import CC359


def get_dataset(img_root_dir, gt_root_dir, slice_axis):
    file_ids = [file_name.split('.')[0] for file_name in listdir(img_root_dir) if not file_name.startswith('.')]
    dataset = CC359(img_root_dir=img_root_dir, gt_root_dir=gt_root_dir, slice_axis=slice_axis,
                    slice_filter_fn=mt_filters.SliceFilter(), file_ids=file_ids)
    return dataset


def load_dataset(dataset, batch_size, num_workers):
    # data augmentation
    transform = tv.transforms.Compose([
        mt_transforms.CenterCrop2D(get_min_max_brain_interval),
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

    data_loader = DataLoader(dataset, batch_size=batch_size,
                             shuffle=True, drop_last=True,
                             num_workers=num_workers,
                             collate_fn=mt_datasets.mt_collate,
                             pin_memory=True)

    return data_loader
