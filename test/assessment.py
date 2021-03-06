import argparse
import os.path

import nibabel as nib
import numpy as np
import pandas as pd

from metrics.dice import dice_score
from metrics.hd import hd_95_rob
import torch

def write_results(col, col_name, file_path):
    csv_input = pd.read_csv(file_path) if os.path.isfile(file_path) else pd.DataFrame()
    csv_input[col_name] = col
    csv_input.to_csv(file_path, index=False)


def assess(files, preds_path, gts_path, mask_type, experiment_name, out_dir):
    mask_suffix = 'pveseg' if mask_type == 'wgc' else 'staple'
    num_labels = 4 if mask_type == 'wgc' else 2
    with open(files) as f:
        file_ids = [line.rstrip() for line in f]

    hds = []
    dice = []
    for file_id in file_ids:
        pred_path = os.path.join(preds_path, '{}_pred.nii.gz'.format(file_id))
        nifti_pred = nib.load(pred_path)
        pred = nifti_pred.get_fdata(dtype=np.float32)

        gt_path = os.path.join(gts_path, '{}_{}.nii.gz'.format(file_id, mask_suffix)) if mask_type != 'wgc' else os.path.join(gts_path, '{}.nii.gz'.format(file_id))
        nifti_gt = nib.load(gt_path)
        gt = nifti_gt.get_fdata(dtype=np.float32)
        print("shapes:", np.shape(nifti_gt), np.shape(pred))
        if num_labels > 2:
            dice_ = 0
            hd_ = 0
            one_hot_mask = torch.nn.functional.one_hot(torch.from_numpy(gt).long(), num_classes=num_labels)
            one_hot_pred = torch.nn.functional.one_hot(torch.from_numpy(pred).long(), num_classes=num_labels)
            gt_data_gpu = one_hot_mask.cuda().float()
            for i in range(num_labels):
                dice_ += dice_score(one_hot_pred, one_hot_mask, num_labels)
            dice_ = dice_/4
        else:
            dice_ = dice_score(pred, gt, num_labels)
        dice.append(dice_)
        hd_95 = hd_95_rob(pred_path, gt_path, num_labels)
        hds.append(hd_95)
    write_results(dice, experiment_name, os.path.join(out_dir, 'dice.csv'))
    write_results(hds, experiment_name, os.path.join(out_dir, 'hd95.csv'))
    write_results([np.mean(dice)], experiment_name, os.path.join(out_dir, 'dice_mean.csv'))
    write_results([np.mean(hds)], experiment_name, os.path.join(out_dir, 'hd95_mean.csv'))
    write_results([np.std(dice)], experiment_name, os.path.join(out_dir, 'dice_std.csv'))
    write_results([np.std(hds)], experiment_name, os.path.join(out_dir, 'hd95_std.csv'))

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-experiment_name', type=str, help='experiment name')
    parser.add_argument('-files', type=str, help='file ids')
    parser.add_argument('-pred', type=str, help='predictions directory path')
    parser.add_argument('-gt', type=str, help='ground truth directory path')
    parser.add_argument('-type', type=str, default='ss', help='mask type')
    parser.add_argument('-out_dir', type=str, default='.', help='output directory')
    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    opt = create_parser()
    assess(opt.files, opt.pred, opt.gt, opt.type, opt.experiment_name, opt.out_dir)
