import argparse

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-data', type=str, help='data path')
    parser.add_argument('-task', type=str, default='ss', help='segmentation task')
    parser.add_argument('-alpha', type=float, default=0.05, help='alpha')
    options = parser.parse_args()
    return options


if __name__ == '__main__':
    opt = create_parser()
    metrics = pd.read_csv(opt.data)
    alpha = opt.alpha
    task = opt.task
    domains = ['ge3', 'ph3', 'si3']
    methods = ['ftl', 'ftf', 'se']
    p_values = np.ones((len(methods), 6))
    headers = []
    for k in range(len(methods)):
        c = 0
        for i in range(len(domains)):
            for j in range(len(domains)):
                if i == j:
                    continue
                method = methods[k]
                source = domains[i]
                target = domains[j]
                base_col = 'test_base_{}_{}_{}'.format(source, target, task)
                col = 'test_{}_{}_{}_{}'.format(method, source, target, task)
                _, p_values[k][c] = wilcoxon(metrics[base_col], metrics[col])
                c += 1
                if i == 0:
                    headers.append('{}_{}'.format(source, target))

    print(np.sum(p_values < alpha))
    df = pd.DataFrame(p_values, columns=headers, index=methods)
    print(df)
