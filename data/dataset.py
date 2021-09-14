import os

from medicaltorch.datasets import MRI2DSegmentationDataset


class CC359(MRI2DSegmentationDataset):
    """This is the ?? dataset.

    :param root_dir: the directory containing the training dataset.
    :param site_ids: a list of site ids to filter (i.e. ['philips', 'ge']).
    :param transform: the transformations that should be applied.
    :param cache: if the data should be cached in memory or not.
    :param slice_axis: axis to make the slicing (default axial).

    .. note:: This dataset assumes that you only have one class in your
              ground truth mask (w/ 0's and 1's). It also doesn't
              automatically resample the dataset.

    .. seealso::
        Prados, F., et al (2017). Spinal cord grey matter
        segmentation challenge. NeuroImage, 152, 312–329.
        https://doi.org/10.1016/j.neuroimage.2017.03.010

        Challenge Website:
        http://cmictig.cs.ucl.ac.uk/spinal-cord-grey-matter-segmentation-challenge
    """
    NUM_SITES = 6
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
            img_filename = self._build_train_input_filename(file_id)
            gt_filename = self._build_train_input_filename(file_id, True)

            img_filename = os.path.join(self.img_root_dir, img_filename)
            gt_filename = os.path.join(self.gt_root_dir, gt_filename)

            if not self.labeled:
                gt_filename = None

            self.filename_pairs.append((img_filename, gt_filename))

        super().__init__(self.filename_pairs, slice_axis, cache,
                         transform, slice_filter_fn, canonical)

    @staticmethod
    def _build_train_input_filename(file_id, mask=False):
        if not mask:
            return "{id}.nii.gz".format(id=file_id)
        else:
            return "{id}_ss.nii.gz".format(id=file_id)
