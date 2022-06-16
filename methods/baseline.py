import os.path
import os.path
import time

import albumentations as A
import torch
from albumentations.pytorch import ToTensorV2
from tensorboardX import SummaryWriter
from tqdm import *

from config.io import MODEL_PATH
from config.param import LAMBDA
from data_data.utils import get_dataloader
from metrics.dice import dice_score
from models.baseline import Unet
from models.utils import EarlyStopping, scheduler
from models.utils import validation


def cmd_train(opt):
    experiment_name = opt["experiment_name"]
    train_dir = opt["train_dir"]
    val_dir = opt["val_dir"]
    if torch.cuda.is_available():
        print('cuda is available')
        torch.cuda.set_device("cuda:0")

    lr = opt["initial_lr"]

    patch_size = (opt["patch_height"], opt["patch_width"])

    train_transform = A.Compose(
        [
            A.ShiftScaleRotate(),
            A.RandomCrop(height=patch_size[0], width=patch_size[1]),
            ToTensorV2(),
        ]
    )

    val_transform = A.Compose(
        [
            A.CenterCrop(height=patch_size[0], width=patch_size[1]),
            ToTensorV2(),
        ]
    )
    problem = opt["problem"]
    img_pth = 'images_wgc' if problem == 'wgc' else 'images'
    msk_pth = 'masks_wgc' if problem == 'wgc' else 'masks'
    train_dataloader = get_dataloader(os.path.join(train_dir, img_pth), os.path.join(train_dir, msk_pth),
                                      opt["batch_size"], train_transform)
    validation_dataloader = get_dataloader(os.path.join(val_dir, img_pth), os.path.join(val_dir, msk_pth),
                                           opt["batch_size"], val_transform)
    out_channels = 4 if problem == 'wgc' else 1
    model = Unet(drop_rate=opt["drop_rate"], out_channels=out_channels)
    model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), weight_decay=LAMBDA, lr=opt["initial_lr"])
    num_epochs = opt["num_epochs"]
    early_stopping = EarlyStopping(opt["patience"], 0.002)

    writer = SummaryWriter(log_dir="log_{}".format(opt["experiment_name"]))

    for epoch in tqdm(range(1, num_epochs + 1), desc="Epochs"):
        start_time = time.time()

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
            if problem == 'wgc':
                one_hot_mask = torch.nn.functional.one_hot(train_mask.long(), num_classes=4).squeeze(-1)
                train_mask = one_hot_mask.cuda().float()
                prediction = model(train_image)
                loss = 0
                for k in range(out_channels):
                    dice_loss = -dice_score(prediction[:, k, :, :], train_mask[:, :, :, k])
                    loss += dice_loss
                loss = loss / out_channels
            else:
                train_mask = train_mask.cuda()
                prediction = model(train_image)
                loss = -dice_score(prediction, train_mask)
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            loss_total += loss.item()
            num_steps += 1

        loss_avg = loss_total / num_steps

        tqdm.write("Steps p/ Epoch: {}".format(num_steps))
        tqdm.write("Train Loss: {:.6f}".format(loss_avg))

        writer.add_scalars('losses', {'loss': loss_avg}, epoch)

        model.eval()

        val_loss = validation(model, validation_dataloader, writer, epoch, out_channels)
        tqdm.write("Validation Loss: {:.6f}".format(val_loss))
        early_stop = early_stopping(val_loss)
        torch.save(model, MODEL_PATH.format(model_name='{}'.format(experiment_name)))

        end_time = time.time()
        total_time = end_time - start_time
        tqdm.write("Epoch {} took {:.2f} seconds.".format(epoch, total_time))
        if early_stop:
            break
