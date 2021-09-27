import argparse


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-batch_size', type=int, default=16, help='input batch size')
    parser.add_argument('-num_workers', type=int, default=16, help='number of workers')
    parser.add_argument('-num_epochs', type=int, default=1, help='number of epochs to train for')
    parser.add_argument('-experiment_name', type=str, default='', help='experiment name')
    parser.add_argument('-lr', type=float, default=1e-3, help='learning rate of the optimizer')
    parser.add_argument('-decay', type=float, default=0.995, help='learning rate of the optimizer')
    parser.add_argument('-validation_split', type=float, default=0.1, help='validation split for training')
    parser.add_argument('-patience', type=int, default=10, help='early stopping patience')
    parser.add_argument('-consistency_loss', type=int, default=3, help='consistency loss')

    opt = parser.parse_args()
    return opt


def train(opt):
    pass


if __name__ == '__main__':
    options = create_parser()
    train(options)
