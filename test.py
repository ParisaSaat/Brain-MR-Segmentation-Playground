import argparse
import os.path
import time
from os import makedirs

import albumentations as A
import medicaltorch.metrics as mt_metrics
import torch
from albumentations.pytorch import ToTensorV2
from tensorboardX import SummaryWriter
from tqdm import *

from config.io import *
from data.utils import get_dataloader
from models.utils import validation


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-model_name', type=str, default='baseline', help='model to load for inference')
    parser.add_argument('-experiment_name', type=str, default='baseline_test', help='experiment name')
    parser.add_argument('-data_dir', type=str, default='', help='test data directory')
    opt = parser.parse_args()
    return opt


def main(opt):
    data_dir = opt.data_dir
    if torch.cuda.is_available():
        torch.cuda.set_device("cuda:0")
    start_time = time.time()

    test_transform = A.Compose(
        [
            ToTensorV2(),
        ]
    )
    writer = SummaryWriter(log_dir="log_test_{}".format(opt.experiment_name))

    test_dataloader = get_dataloader(os.path.join(data_dir, 'images'), os.path.join(data_dir, 'masks'), 16,
                                     test_transform)
    metric_fns = [mt_metrics.dice_score, mt_metrics.jaccard_score, mt_metrics.hausdorff_score,
                  mt_metrics.precision_score, mt_metrics.recall_score,
                  mt_metrics.specificity_score, mt_metrics.intersection_over_union,
                  mt_metrics.accuracy_score]

    model = torch.load(MODEL_PATH.format(model_name=opt.model_name))
    test_samples_dir = 'test_samples_{}'.format(opt.experiment_name)
    makedirs(test_samples_dir)
    test_loss = validation(model, test_dataloader, writer, metric_fns, 0, test_samples_dir)
    tqdm.write("Validation Loss: {:.6f}".format(test_loss))

    end_time = time.time()
    total_time = end_time - start_time
    tqdm.write("Testing took {:.2f} seconds.".format(total_time))


if __name__ == '__main__':
    options = create_parser()
    main(options)
