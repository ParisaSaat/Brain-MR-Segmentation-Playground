import argparse

import numpy as np
from scipy.stats import wilcoxon


def wilcoxon_test(data1, data2, alpha):
    stat, p = wilcoxon(data1, data2)
    print('Statistics=%.3f, p=%.3f' % (stat, p))

    if p > alpha:
        print('Same distribution (fail to reject H0)')
    else:
        print('Different distribution (reject H0)')


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-x_path', type=str, help='data1 path')
    parser.add_argument('-y_path', type=str, help='data2 path')
    parser.add_argument('-alpha', type=float, default=0.05, help='alpha')
    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    opt = create_parser()
    x = np.load(opt.x_path)
    y = np.load(opt.y_path)
    wilcoxon_test(x, y, opt.alpha)
