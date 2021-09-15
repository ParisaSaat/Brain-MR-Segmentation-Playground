import argparse, time

from data.utils import load_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-batch_size', type=int, default=16, help='input batch size')
    parser.add_argument('-num_workers', type=int, default=16, help='number of workers')
    parser.add_argument('-num_epochs', type=int, default=1, help='number of epochs to train for')
    parser.add_argument('-experiment_name', type=str, default='', help='experiment name')
    parser.add_argument('-lr', type=float, default=1e-3, help='learning rate of the optimizer')
    parser.add_argument('-decay', type=float, default=0.995, help='learning rate of the optimizer')
    parser.add_argument('-validation_split', type=float, default=0.1, help='validation split for training')
    parser.add_argument('-patience', type=int, default=10, help='early stopping patience')
    parser.add_argument('-patch_size_row', type=int, default='128', help='patch width')
    parser.add_argument('-patch_size_col', type=int, default='128', help='patch height')
    parser.add_argument('-max_patches', type=int, default=3, help='number of patches per slice')
    parser.add_argument('-consistency_loss', type=int, default=3, help='consistency loss')
    parser.add_argument('-imgs_dir', type=str, default='imgs_train',
                        help='images directory name')
    parser.add_argument('-masks_dir', type=str, default='imgs_masks_train',
                        help='mask directory name')

    opt = parser.parse_args()

    patch_size = opt.patch_size_row, opt.patch_size_col

    start_time = time.time()

    tags = [0, 1, 2]

    for tt in tags:
        load_dataset(opt.imgs_dir, opt.masks_dir, tt, opt.batch_size, opt.num_workers)

    print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == '__main__':
    main()
