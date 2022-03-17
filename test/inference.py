import argparse
import os.path
import time

import nibabel as nib
import numpy as np
import torch
from tqdm import *

from config.io import *
from models.baseline import Unet
from models.unlearn import Segmenter

MODELS = {
    'baseline': Unet(),
    'unlearn': Segmenter()
}


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-model_name', type=str, default='baseline', help='model to load for inference')
    parser.add_argument('-experiment_name', type=str, default='baseline_test', help='experiment name')
    parser.add_argument('-data_dir', type=str, default='', help='test data directory')
    parser.add_argument('-pred_dir', type=str, default='', help='predictions directory')
    parser.add_argument('-problem', type=str, default='skull-stripping', help='segmentation problem')
    parser.add_argument('-state_dict', type=str, default=None, help='model state dict path')
    parser.add_argument('-files', type=str, help='file ids')
    opt = parser.parse_args()
    return opt


def load_pretrained_model(model_name, path):
    model = MODELS[model_name]
    model.load_state_dict(torch.load(path))
    return model


def main(opt):
    data_dir = opt.data_dir
    if torch.cuda.is_available():
        torch.cuda.set_device("cuda:0")
    images_path = 'images_wgc' if opt.problem == 'wgc' else 'images'
    masks_path = 'masks_wgc' if opt.problem == 'wgc' else 'masks'
    mask_type = 'pveseg' if opt.problem == 'wgc' else 'staple'
    start_time = time.time()
    if opt.state_dict:
        model = load_pretrained_model(opt.model_name, opt.state_dict)
    else:
        model = torch.load(MODEL_PATH.format(model_name=opt.model_name))
    model.eval()

    with open(opt.files) as f:
        file_ids = [line.rstrip() for line in f]
    with torch.no_grad():
        for file_id in file_ids:
            img_path = os.path.join(data_dir, '{}/{}.nii.gz'.format(images_path, file_id))
            nifti_image = nib.load(img_path)
            image = nifti_image.get_fdata(dtype=np.float32)
            image_data_gpu = torch.tensor(image).cuda()
            mask_path = os.path.join(data_dir, '{}/{}_{}.nii.gz'.format(masks_path, file_id, mask_type))
            nifti_mask = nib.load(mask_path)
            mask_affine = nifti_mask.affine
            output_volume = np.zeros(image_data_gpu.shape)
            for i in range(image_data_gpu.shape[0]):
                model_out = model(image_data_gpu[i].unsqueeze(0).unsqueeze(0))
                decoded = torch.argmax(model_out, dim=1) if opt.problem == 'wgc' else model_out
                output_volume[i] = decoded.cpu()
            pred = nib.Nifti1Image(output_volume, affine=mask_affine)
            nib.save(pred, os.path.join(opt.pred_dir, '{}_pred.nii.gz'.format(file_id)))
    end_time = time.time()
    total_time = end_time - start_time
    tqdm.write("Testing took {:.2f} seconds.".format(total_time))


if __name__ == '__main__':
    options = create_parser()
    main(options)
