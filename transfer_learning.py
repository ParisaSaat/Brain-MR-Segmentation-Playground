import argparse
import time

import albumentations as A
import medicaltorch.losses as mt_losses
import torch
from albumentations.pytorch import ToTensorV2
from tensorboardX import SummaryWriter
from tqdm import *

from config.io import *
from config.param import *
from train_baseline import get_dataloader
from train_baseline import scheduler


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-batch_size', type=int, default=32, help='input batch size')
    parser.add_argument('-data_dir', type=str, help='images directory name')
    parser.add_argument('-model_name', type=str, default='baseline', help='model to load for inference')
    parser.add_argument('-experiment_name', type=str, default='tl', help='experiment name')
    parser.add_argument('-num_epochs', type=int, default=15, help='number of epochs to train for')
    parser.add_argument('-initial_lr', type=float, default=0.000125, help='learning rate of the optimizer')
    parser.add_argument('-patch_height', type=int, default=128, help='patch size')
    parser.add_argument('-patch_width', type=int, default=128, help='patch size')
    opt = parser.parse_args()
    return opt


def tl(opt):
    num_epochs = opt.num_epochs
    if torch.cuda.is_available():
        torch.cuda.set_device("cuda:0")
    start_time = time.time()
    patch_size = (opt.patch_height, opt.patch_width)
    data_dir = opt.data_dir

    train_transform = A.Compose(
        [
            A.RandomCrop(height=patch_size[0], width=patch_size[1]),
            ToTensorV2(),
        ]
    )

    train_dataloader = get_dataloader(os.path.join(data_dir, 'images'), os.path.join(data_dir, 'masks'),
                                      opt.batch_size, train_transform)
    tl_size = len(train_dataloader) / 4

    model = torch.load(MODEL_PATH.format(model_name=opt.model_name))
    model.cuda()
    ct = 0
    for child in model.children():
        if ct != 0:
            for param in child.parameters():
                param.requires_grad = False
        ct += 1
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=LAMBDA, lr=opt.initial_lr)
    experiment_name = opt.experiment_name
    writer = SummaryWriter(log_dir="log_{}".format(experiment_name))
    lr = opt.initial_lr
    for epoch in tqdm(range(1, num_epochs + 1), desc="Epochs"):
        batch_cnt = 0
        lr = scheduler(epoch, lr)

        writer.add_scalar('learning_rate', lr, epoch)

        for param_group in optimizer.param_groups:
            tqdm.write("Learning Rate: {:.6f}".format(lr))
            param_group['lr'] = lr

        model.train()

        loss_total = 0.0

        num_steps = 0

        for i, train_batch in enumerate(train_dataloader):
            train_image, train_mask = train_batch['image'], train_batch['mask']
            train_image = train_image.cuda()
            train_mask = train_mask.cuda()
            prediction = model(train_image)
            loss = mt_losses.dice_loss(prediction, train_mask)
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            loss_total += loss.item()
            num_steps += 1
            batch_cnt += 1
            if batch_cnt > tl_size:
                break

        loss_avg = loss_total / num_steps

        tqdm.write("Steps p/ Epoch: {}".format(num_steps))
        tqdm.write("Train Loss: {:.6f}".format(loss_avg))

        writer.add_scalars('losses', {'loss': loss_avg}, epoch)

        torch.save(model, MODEL_PATH.format(model_name='{}'.format(experiment_name)))

        end_time = time.time()
        total_time = end_time - start_time
        tqdm.write("Epoch {} took {:.2f} seconds.".format(epoch, total_time))


if __name__ == '__main__':
    options = create_parser()
    tl(options)
