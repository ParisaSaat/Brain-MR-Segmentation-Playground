import medicaltorch.datasets as mt_datasets
import medicaltorch.filters as mt_filters
import medicaltorch.transforms as mt_transforms
import torchvision as tv
from torch.utils.data import DataLoader
import numpy as np
from os import listdir

from data.dataset import CC359


''' Get interval where data is non-zero in the volume'''


def get_min_max_brain_interval(img_slice):
    min_idx = max_idx = np.asarray([])
    check = aux = i = 0

    if img_slice.sum() == 0:
        if aux != 0:
            max_idx = aux - 1
        i += 1
    else:
        if aux == 0:
            min_idx = i
            aux = i
        aux += 1
    print(min_idx, max_idx)
    return min_idx, max_idx


def load_dataset(img_root_dir, gt_root_dir, slice_axis, batch_size, num_workers):
    file_ids = [file_name.split('.')[0] for file_name in listdir(img_root_dir) if not file_name.startswith('.')]
    dataset = CC359(img_root_dir=img_root_dir, gt_root_dir=gt_root_dir, slice_axis=slice_axis,
                    slice_filter_fn=mt_filters.SliceFilter(), file_ids=file_ids)

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
