import argparse
import os.path

import nibabel as nib
import numpy as np

from metrics.dice import dice_score
from metrics.hd import hausdorff_score


def assess(files, preds_path, gts_path, mask_type):
    mask_suffix = 'pveseg' if mask_type == 'wgc' else 'staple'
    with open(files) as f:
        file_ids = [line.rstrip() for line in f]

    hd = []
    dice = []
    for file_id in file_ids:
        pred_path = os.path.join(preds_path, '{}_{}.nii.gz'.format(file_id, mask_suffix))
        nifti_pred = nib.load(pred_path)
        pred = nifti_pred.get_fdata(dtype=np.float32)

        gt_path = os.path.join(gts_path, '{}_{}.nii.gz'.format(file_id, mask_suffix))
        nifti_gt = nib.load(gt_path)
        gt = nifti_gt.get_fdata(dtype=np.float32)

        dice.append(dice_score(pred, gt))
        hd.append(hausdorff_score(pred, gt))

    results = {
        'dice_score': sum(dice) / len(dice),
        'hausdorff_score': sum(hd) / len(hd)
    }
    return results


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-files', type=str, help='file ids')
    parser.add_argument('-pred', type=str, help='predictions directory path')
    parser.add_argument('-gt', type=str, help='ground truth directory path')
    parser.add_argument('-type', type=str, default='ss', help='mask type')
    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    opt = create_parser()
    result = assess(opt.files, opt.pred, opt.gt, opt.type)
    print(result)
