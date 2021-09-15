import os

from medicaltorch.datasets import MRI2DSegmentationDataset


class CC359(MRI2DSegmentationDataset):
    """This is the CC359 dataset.

    :param img_root_dir: the directory containing the MRI images.
    :param gt_root_dir: the directory containing the ground truth masks.
    :param transform: the transformations that should be applied.
    :param cache: if the data should be cached in memory or not.
    :param slice_axis: axis to make the slicing (default axial).
    """
    NUM_SITES = 2
    NUM_SUBJECTS = 359

    def __init__(self, img_root_dir, slice_axis, gt_root_dir=None, file_ids=None, cache=True, transform=None,
                 slice_filter_fn=None,
                 canonical=False, labeled=True):

        self.labeled = labeled
        self.img_root_dir = img_root_dir
        self.gt_root_dir = gt_root_dir
        self.file_ids = file_ids

        self.filename_pairs = []

        for file_id in self.file_ids:
            img_filename = self._build_input_filename(file_id)
            gt_filename = self._build_input_filename(file_id, True)

            img_filename = os.path.join(self.img_root_dir, img_filename)
            gt_filename = os.path.join(self.gt_root_dir, gt_filename)

            if not self.labeled:
                gt_filename = None

            self.filename_pairs.append((img_filename, gt_filename))

        super().__init__(self.filename_pairs, slice_axis, cache,
                         transform, slice_filter_fn, canonical)

    @staticmethod
    def _build_input_filename(file_id, mask=False):
        if not mask:
            return "{id}.nii.gz".format(id=file_id)
        else:
            return "{id}_ss.nii.gz".format(id=file_id)
