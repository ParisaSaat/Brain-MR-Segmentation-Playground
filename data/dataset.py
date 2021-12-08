import os
from os import makedirs

import nibabel as nib
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from config.param import SLICE_HEIGHT, SLICE_WIDTH


class SliceFilter(object):

    def __init__(self, filter_empty_mask=False,
                 filter_empty_input=True):
        self.filter_empty_mask = filter_empty_mask
        self.filter_empty_input = filter_empty_input

    def __call__(self, sample):
        input_data, gt_data = sample['input'], sample['gt']

        if self.filter_empty_mask:
            if not np.any(gt_data):
                return False

        if self.filter_empty_input:
            if not np.any(input_data):
                return False

        return True


class SegmentationPair2D(object):
    """This class is used to build 2D segmentation datasets. It represents
    a pair of of two data volumes (the input data and the ground truth data).

    :param input_filename: the input filename (supported by nibabel).
    :param gt_filename: the ground-truth filename.
    :param cache: if the data should be cached in memory or not.
    :param canonical: canonical reordering of the volume axes.
    """

    def __init__(self, input_filename, gt_filename, normalizer=None):
        self.input_filename = input_filename
        self.gt_filename = gt_filename
        self.normalizer = normalizer

        self.input_handle = nib.load(self.input_filename)
        self.input_affine = self.input_handle.affine

        # Unlabeled data (inference time)
        if self.gt_filename is None:
            self.gt_handle = None
            self.gt_affine = None
        else:
            self.gt_handle = nib.load(self.gt_filename)
            self.gt_affine = self.gt_handle.affine

        if len(self.input_handle.shape) > 3:
            raise RuntimeError("4-dimensional volumes not supported.")

        # Sanity check for dimensions, should be the same
        input_shape, gt_shape = self.get_pair_shapes()

        if self.gt_handle is not None:
            if not np.allclose(input_shape, gt_shape):
                raise RuntimeError('Input and ground truth with different dimensions.')

            # Unlabeled data
            if self.gt_handle is not None:
                self.gt_handle = nib.as_closest_canonical(self.gt_handle)

    def get_pair_shapes(self):
        """Return the tuple (input, ground truth) representing both the input
        and ground truth shapes."""
        input_shape = self.input_handle.header.get_data_shape()

        # Handle unlabeled data
        if self.gt_handle is None:
            gt_shape = None
        else:
            gt_shape = self.gt_handle.header.get_data_shape()

        return input_shape, gt_shape

    def get_pair_data(self):
        """Return the tuble (input, ground truth) with the data content in
        numpy array."""
        input_data = self.input_handle.get_fdata(dtype=np.float32)

        # Handle unlabeled data
        if self.gt_handle is None:
            gt_data = None
        else:
            gt_data = self.gt_handle.get_fdata(dtype=np.uint8)

        if self.normalizer:
            input_data = self.normalizer(input_data)
            if self.gt_handle is not None:
                gt_data = self.normalizer(gt_data)

        return input_data, gt_data

    def get_pair_slice(self, slice_index, slice_axis=2):
        """Return the specified slice from (input, ground truth).

        :param slice_index: the slice number.
        :param slice_axis: axis to make the slicing.
        """
        # use dataobj to avoid caching
        input_dataobj = self.input_handle.dataobj

        if self.gt_handle is None:
            gt_dataobj = None
        else:
            gt_dataobj = self.gt_handle.dataobj

        if slice_axis not in [0, 1, 2]:
            raise RuntimeError("Invalid axis, must be between 0 and 2.")

        if slice_axis == 2:
            input_slice = np.asarray(input_dataobj[..., slice_index],
                                     dtype=np.float32)
        elif slice_axis == 1:
            input_slice = np.asarray(input_dataobj[:, slice_index, ...],
                                     dtype=np.float32)
        elif slice_axis == 0:
            input_slice = np.asarray(input_dataobj[slice_index, ...],
                                     dtype=np.float32)

        # Handle the case for unlabeled data
        if self.gt_handle is None:
            gt_slice = None
        else:
            if slice_axis == 2:
                gt_slice = np.asarray(gt_dataobj[..., slice_index],
                                      dtype=np.float32)
            elif slice_axis == 1:
                gt_slice = np.asarray(gt_dataobj[:, slice_index, ...],
                                      dtype=np.float32)
            elif slice_axis == 0:
                gt_slice = np.asarray(gt_dataobj[slice_index, ...],
                                      dtype=np.float32)

        dreturn = {
            "input": input_slice,
            "gt": gt_slice,
            "input_affine": self.input_affine,
            "gt_affine": self.gt_affine,
        }

        return dreturn


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

    def __init__(self, img_root_dir, gt_root_dir=None, slice_axis=1, file_ids=None, cache=True, transform=None,
                 slice_filter_fn=SliceFilter, canonical=False, labeled=True, normalizer=None, mask_type='staple'):

        self.labeled = labeled
        self.img_root_dir = img_root_dir
        self.gt_root_dir = gt_root_dir
        self.file_ids = file_ids
        self.normalizer = normalizer
        self.mask_type = mask_type

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

    def _load_filenames(self):
        for input_filename, gt_filename in self.filename_pairs:
            seg_pair = SegmentationPair2D(input_filename, gt_filename, self.normalizer)
            self.handlers.append(seg_pair)

    def save_slices(self, image_path, mask_path):
        makedirs(image_path, exist_ok=True)
        makedirs(mask_path, exist_ok=True)
        n_slices = 0
        for seg_pair in self.handlers:
            input_data_shape, _ = seg_pair.get_pair_shapes()
            for seg_pair_slice in range(input_data_shape[self.slice_axis]):
                slice_pair = seg_pair.get_pair_slice(seg_pair_slice, self.slice_axis)
                if self.slice_filter_fn:
                    filter_fn_ret = self.slice_filter_fn(slice_pair)
                    if not filter_fn_ret:
                        continue
                image = slice_pair.get("input")
                mask = slice_pair.get("gt")
                resized_img = np.zeros((SLICE_HEIGHT, SLICE_WIDTH))
                resized_mask = np.zeros((SLICE_HEIGHT, SLICE_WIDTH))
                resized_img[:image.shape[0], :image.shape[1]] = image
                resized_mask[:mask.shape[0], :mask.shape[1]] = mask
                nifti_image = nib.Nifti1Image(resized_img, affine=slice_pair.get("input_affine"))
                nifti_mask = nib.Nifti1Image(resized_mask, slice_pair.get("gt_affine"))
                nib.save(nifti_image, os.path.join(image_path, '{}.nii'.format(n_slices)))
                nib.save(nifti_mask, os.path.join(mask_path, '{}.nii'.format(n_slices)))
                n_slices += 1
        print("total number of slices:", n_slices)

    def __len__(self):
        """Return the dataset size."""
        return len(self.indexes)

    def __getitem__(self, index):
        """Return the specific index pair slices (input, ground truth).

        :param index: slice index.
        """
        seg_pair, seg_pair_slice = self.indexes[index]
        pair_slice = seg_pair.get_pair_slice(seg_pair_slice,
                                             self.slice_axis)

        # Consistency with torchvision, returning PIL Image
        # Using the "Float mode" of PIL, the only mode
        # supporting unbounded float32 values
        input_img = np.array(Image.fromarray(pair_slice["input"], mode='F'))

        # Handle unlabeled data
        if pair_slice["gt"] is None:
            gt_img = None
        else:
            gt_img = np.array(Image.fromarray(pair_slice["gt"], mode='F'))

        if self.transform is not None:
            transformed_pair = self.transform(image=input_img, mask=gt_img)
            input_img = transformed_pair['image']
            gt_img = transformed_pair['mask']

        data_dict = {
            'input': input_img,
            'gt': gt_img,
        }
        return data_dict

    def _build_input_filename(self, file_id, mask=False):
        if not mask:
            return "{id}.nii.gz".format(id=file_id)
        else:
            return "{id}_{mask_type}.nii.gz".format(id=file_id, mask_type=self.mask_type)


class BrainMRI2D(Dataset):
    def __init__(self, img_root_dir, gt_root_dir=None, file_ids=None, transform=None, labeled=True, mean_teacher=False):
        self.img_root_dir = img_root_dir
        self.gt_root_dir = gt_root_dir
        self.file_ids = file_ids
        self.transform = transform
        self.labeled = labeled
        self.mean_teacher = mean_teacher

        self.pairs_path = []
        for file_id in self.file_ids:
            img_path = os.path.join(self.img_root_dir, file_id)
            gt_path = os.path.join(self.gt_root_dir, file_id) if self.labeled else None
            self.pairs_path.append((img_path, gt_path))

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs_path[idx]
        nifti_image = nib.load(img_path)
        image_affine = nifti_image.affine
        image = nifti_image.get_fdata(dtype=np.float32)
        nifti_mask = nib.load(mask_path)
        mask = nifti_mask.get_fdata(dtype=np.float32)
        mask_affine = nifti_mask.affine

        if self.transform:
            if self.mean_teacher:
                transformed = self.transform({'input': Image.fromarray(image), 'gt': Image.fromarray(mask)})
                image = transformed.get('input')
                mask = transformed.get('gt')
            else:
                transformed = self.transform(image=image, mask=mask)
                image = transformed.get('image')
                mask = transformed.get('mask')
        sample = {'image': image, 'mask': mask, 'mask_affine': mask_affine, 'image_affine': image_affine, 'idx': idx}
        return sample

    def __len__(self):
        return len(self.pairs_path)
