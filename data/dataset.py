import os

import numpy as np
from PIL import Image
from medicaltorch.datasets import SegmentationPair2D
from torch.utils.data import Dataset


class SegmentationPair(SegmentationPair2D):

    def __init__(self, input_filename, gt_filename, normalizer=None, cache=True, canonical=False):
        super().__init__(input_filename, gt_filename, cache, canonical)
        self.normalizer = normalizer

    def get_pair_data(self):
        """Return the tuble (input, ground truth) with the data content in
        numpy array."""
        cache_mode = 'fill' if self.cache else 'unchanged'
        input_data = self.input_handle.get_fdata(cache_mode, dtype=np.float32)

        # Handle unlabeled data
        if self.gt_handle is None:
            gt_data = None
        else:
            gt_data = self.gt_handle.get_fdata(cache_mode, dtype=np.float32)

        if self.normalizer:
            input_data = self.normalizer(input_data)
            if self.gt_handle is not None:
                gt_data = self.normalizer(gt_data)

        return input_data, gt_data


class CC359(Dataset):
    """This is the CC359 dataset.

    :param img_root_dir: the directory containing the MRI images.
    :param gt_root_dir: the directory containing the ground truth masks.
    :param transform: the transformations that should be applied.
    :param cache: if the data should be cached in memory or not.
    :param slice_axis: axis to make the slicing (default axial).
    """
    NUM_SITES = 2
    NUM_SUBJECTS = 359

    def __init__(self, img_root_dir, slice_axis, normalizer, gt_root_dir=None, file_ids=None, cache=True,
                 transform=None, slice_filter_fn=None, canonical=False, labeled=True):

        self.labeled = labeled
        self.img_root_dir = img_root_dir
        self.gt_root_dir = gt_root_dir
        self.file_ids = file_ids
        self.normalizer = normalizer

        self.filename_pairs = []

        for file_id in self.file_ids:
            img_filename = self._build_input_filename(file_id)
            gt_filename = self._build_input_filename(file_id, True)

            img_filename = os.path.join(self.img_root_dir, img_filename)
            gt_filename = os.path.join(self.gt_root_dir, gt_filename)

            if not self.labeled:
                gt_filename = None

            self.filename_pairs.append((img_filename, gt_filename))


        self.filename_pairs = self.filename_pairs
        self.handlers = []
        self.indexes = []
        self.transform = transform
        self.cache = cache
        self.slice_axis = slice_axis
        self.slice_filter_fn = slice_filter_fn
        self.canonical = canonical

        self._load_filenames()
        self._prepare_indexes()

    def _load_filenames(self):
        for input_filename, gt_filename in self.filename_pairs:
            segpair = SegmentationPair(input_filename, gt_filename, self.normalizer, self.cache, self.canonical)
            self.handlers.append(segpair)

    def _prepare_indexes(self):
        for segpair in self.handlers:
            input_data_shape, _ = segpair.get_pair_shapes()
            for segpair_slice in range(input_data_shape[self.slice_axis]):

                # Check if slice pair should be used or not
                if self.slice_filter_fn:
                    slice_pair = segpair.get_pair_slice(segpair_slice,
                                                        self.slice_axis)

                    filter_fn_ret = self.slice_filter_fn(slice_pair)
                    if not filter_fn_ret:
                        continue

                item = (segpair, segpair_slice)
                self.indexes.append(item)

    def set_transform(self, transform):
        """This method will replace the current transformation for the
        dataset.

        :param transform: the new transformation
        """
        self.transform = transform

    def __len__(self):
        """Return the dataset size."""
        return len(self.indexes)

    def __getitem__(self, index):
        """Return the specific index pair slices (input, ground truth).

        :param index: slice index.
        """
        segpair, segpair_slice = self.indexes[index]
        pair_slice = segpair.get_pair_slice(segpair_slice,
                                            self.slice_axis)

        # Consistency with torchvision, returning PIL Image
        # Using the "Float mode" of PIL, the only mode
        # supporting unbounded float32 values
        input_img = Image.fromarray(pair_slice["input"], mode='F')

        # Handle unlabeled data
        if pair_slice["gt"] is None:
            gt_img = None
        else:
            gt_img = Image.fromarray(pair_slice["gt"], mode='F')

        data_dict = {
            'input': input_img,
            'gt': gt_img,
        }
        if self.transform is not None:
            data_dict = self.transform(data_dict)
        return data_dict

    @staticmethod
    def _build_input_filename(file_id, mask=False):
        if not mask:
            return "{id}.nii.gz".format(id=file_id)
        else:
            return "{id}_ss.nii.gz".format(id=file_id)
