from functools import partial

import SimpleITK as sitk
import numpy as np
from SimpleITK import GetArrayViewFromImage as ArrayView


def hd_95_rob(pred_path, gt_path, num_labels):
    """

    :param num_labels: Number of labels in the ground truth segmentation mask (we don't count label 0).
    We assume the labels go from zero to num_labels
    :param pred_path: Path to segmentation prediction
    :param gt_path: Path to ground-truth segmentation
    :return: Haussdorff distance 95th percentile
    """

    # Load segmentation masks
    prediction = sitk.ReadImage(pred_path)
    gold = sitk.ReadImage(gt_path)

    # Distance map
    distance_map = partial(sitk.SignedMaurerDistanceMap, squaredDistance=False, useImageSpacing=True)
    hd95 = np.zeros(num_labels-1)
    for label in range(1, num_labels):
        gold_surface = sitk.LabelContour(gold == label, False)
        prediction_surface = sitk.LabelContour(prediction == label, False)

        prediction_distance_map = sitk.Abs(distance_map(prediction_surface))
        gold_distance_map = sitk.Abs(distance_map(gold_surface))

        gold_to_prediction = ArrayView(prediction_distance_map)[ArrayView(gold_surface) == 1]
        prediction_to_gold = ArrayView(gold_distance_map)[ArrayView(prediction_surface) == 1]

        # Find the 95% Distance for each direction and average
        hd95[label - 1] = ((np.percentile(gold_to_prediction, 95) + np.percentile(gold_to_prediction, 95)) / 20)
    return np.mean(hd95)
