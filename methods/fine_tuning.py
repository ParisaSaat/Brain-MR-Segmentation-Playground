import time

import albumentations as A
import torch
from albumentations.pytorch import ToTensorV2
from tensorboardX import SummaryWriter
from tqdm import *

from config.io import *
from config.param import *
from data_data.utils import get_dataloader
from models.utils import dice_score
from models.utils import scheduler


def cmd_train(opt):
    num_epochs = opt["num_epochs"]
    mode = opt["mode"]
    problem = opt["problem"]
    img_pth = 'images_wgc' if problem == 'wgc' else 'images'
    msk_pth = 'masks_wgc' if problem == 'wgc' else 'masks'
    num_labels = 4 if problem == 'wgc' else 2
    out_channels = 4 if problem == 'wgc' else 1
    if torch.cuda.is_available():
        torch.cuda.set_device("cuda:0")
    start_time = time.time()
    patch_size = (opt["patch_height"], opt["patch_width"])
    data_dir = opt["data_dir"]

    train_transform = A.Compose(
        [
            A.RandomCrop(height=patch_size[0], width=patch_size[1]),
            ToTensorV2(),
        ]
    )

    train_dataloader = get_dataloader(os.path.join(data_dir, img_pth), os.path.join(data_dir, msk_pth),
                                      opt["batch_size"], train_transform)
    tl_size = len(train_dataloader) / 4

    model = torch.load(MODEL_PATH.format(model_name=opt["model_name"]))
    model.cuda()
    ct = 0
    for child in model.children():
        if (mode == 'first' and ct != 0) or (mode == 'last' and ct != 9):
            for param in child.parameters():
                param.requires_grad = False
        ct += 1
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=LAMBDA, lr=opt["initial_lr"])
    experiment_name = opt["experiment_name"]
    writer = SummaryWriter(log_dir="log_{}".format(experiment_name))
    lr = opt["initial_lr"]
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
            if problem == 'wgc':
                one_hot_mask = torch.nn.functional.one_hot(train_mask.long(), num_classes=4).squeeze(-1)
                train_mask = one_hot_mask.cuda().float()
                prediction = model(train_image)
                loss = 0
                for k in range(out_channels):
                    loss += dice_score(prediction[:, k, :, :], train_mask[:, :, :, k], num_labels)
                loss = loss / out_channels
            else:
                train_mask = train_mask.cuda()
                prediction = model(train_image)
                loss = dice_score(prediction, train_mask, num_labels)
            optimizer.zero_grad()
            loss = torch.tensor(loss, requires_grad=True)
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
