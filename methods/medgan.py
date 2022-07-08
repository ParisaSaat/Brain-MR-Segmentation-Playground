from email.generator import Generator
import os.path
from pyexpat import features
import time

import albumentations as A
import torch
from albumentations.pytorch import ToTensorV2
from tensorboardX import SummaryWriter
from tqdm import *
import torch.nn as nn

from config.io import MODEL_PATH
from config.param import LAMBDA
from data.utils import get_dataloader
from metrics.dice import dice_score
from models.medgan2 import CasUNet, Discriminator, VGG, UNet
from models.utils import EarlyStopping, scheduler
from models.utils import validation, gram_matrix, medgan_val
from models.baseline import Unet

def cmd_train(opt):
    experiment_name = opt["experiment_name"]
    source_train_dir = opt["source_train_dir"]
    source_val_dir = opt["source_val_dir"]
    target_train_dir = opt["target_train_dir"]
    target_val_dir = opt["target_val_dir"]
    if torch.cuda.is_available():
        print('cuda is available')
        torch.cuda.set_device("cuda:0")

    lr = opt["initial_lr"]
    lambda_L1 = opt['lambda_L1']
    patch_size = (opt["patch_height"], opt["patch_width"])

    train_transform = A.Compose(
        [
            A.ShiftScaleRotate(),
            A.RandomCrop(height=patch_size[0], width=patch_size[1]),
            #A.Resize(256, 256), #, keep 256 from paper or take 288?
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

    source_train_loader = get_dataloader(os.path.join(source_train_dir, img_pth), os.path.join(source_train_dir, msk_pth),
                                      opt["batch_size"], train_transform)
    source_validation_loader = get_dataloader(os.path.join(source_val_dir, img_pth), os.path.join(source_val_dir, msk_pth),
                                           opt["batch_size"], val_transform)
    target_train_loader = get_dataloader(os.path.join(target_train_dir, img_pth), os.path.join(target_train_dir, msk_pth),
                                      opt["batch_size"], train_transform)
    target_validation_loader = get_dataloader(os.path.join(target_val_dir, img_pth), os.path.join(target_val_dir, msk_pth),
                                           opt["batch_size"], val_transform)
    out_channels = 4 if problem == 'wgc' else 1

    generator = CasUNet(n_unet = 2, io_channels = 1)
    generator.cuda()
    generator.train()
    discriminator = Discriminator(input_nc=2)
    discriminator.cuda()
    discriminator.train()
    vgg = VGG().cuda()
    uNet = torch.load(opt["UNET_PATH"]).cuda()
    uNet.eval()

    vgg.eval()

    G_optimizer = torch.optim.Adam(generator.parameters(), lr=opt["initial_lr"])
    D_optimizer = torch.optim.Adam(discriminator.parameters(), lr=opt["initial_lr"])
    num_epochs = opt["num_epochs"]
    # early_stopping = EarlyStopping(opt["patience"], 0.002)

    writer = SummaryWriter(log_dir="log_{}".format(opt["experiment_name"]))
    BCEloss = nn.BCEWithLogitsLoss()
    L1loss = nn.L1Loss()
    MSEloss = nn.MSELoss()

    itr = 0
    for epoch in tqdm(range(1, num_epochs + 1), desc="Epochs"):
        start_time = time.time()

        lr = scheduler(epoch, lr)

        writer.add_scalar('learning_rate', lr, epoch)

        for param_group in G_optimizer.param_groups:
            tqdm.write("Gen Learning Rate: {:.6f}".format(lr))
            param_group['lr'] = lr
        for param_group in D_optimizer.param_groups:
            tqdm.write("Disc Learning Rate: {:.6f}".format(lr))
            param_group['lr'] = lr

        G_loss_epoch = 0.0
        D_loss_epoch = 0.0
        G_loss = 0.0
        D_loss = 0.0
        num_steps = 0
        source_train_iter = iter(source_train_loader)

        for i, target_batch in tqdm(enumerate(target_train_loader)):
            real_tgt = target_batch["image"]
            real_tgt = real_tgt.cuda()
            real_src = source_train_iter.next()["image"]
            real_src = real_src.cuda()

            # train Generator for 3 iterations on same input batch
            for _ in  range(3):
                discriminator.eval()
                fake_src = generator(real_tgt)
                G_optimizer.zero_grad()
                fake_tgt_src = torch.cat((real_tgt, fake_src), 1)
                pred_fake, features1 = discriminator(fake_tgt_src)
                G_loss = BCEloss(pred_fake, torch.ones_like(pred_fake))
                G_loss += L1loss(fake_src, real_src) * lambda_L1
                _, features2 = discriminator(torch.cat((real_tgt, real_src), 1))

                Lpercp = 0.0
                for feat1, feat2 in zip(features1, features2):
                    Lpercp += 0.5 * L1loss(feat1, feat2)


                features_vgg_fake = vgg(torch.cat((fake_src,fake_src,fake_src), 1))
                features_vgg_real = vgg(torch.cat((real_src,real_src,real_src), 1))

                Gl1 = MSEloss(gram_matrix(features_vgg_fake[0]), gram_matrix(features_vgg_real[0]))
                Gl2 = MSEloss(gram_matrix(features_vgg_fake[-1]), gram_matrix(features_vgg_real[-1]))
                Lcontent = 0.0
                Lstyle = 1/4 * (Gl1 + Gl2) * 0.5
                w = 0.1
                for feat1, feat2 in zip(features_vgg_fake, features_vgg_real):
                    Lcontent += w * MSEloss(feat1, feat2)
                    w /= 2

                G_loss += 20 * Lpercp + 0.0001 * Lstyle + 0.0001 * Lcontent
                G_loss_epoch += G_loss
                G_loss.backward()
                G_optimizer.step()

                

            # train Discriminator once for every 3 iterations of Generator
            discriminator.train()
            discriminator.zero_grad()
            fake_tgt_src = torch.cat((real_tgt, generator(real_tgt)), 1)
            pred_fake, _ = discriminator(fake_tgt_src.detach())
            loss_fake = BCEloss(pred_fake, torch.zeros_like(pred_fake))
            real_tgt_src = torch.cat((real_tgt, real_src), 1)
            pred_real, _ = discriminator(real_tgt_src)
            loss_real = BCEloss(pred_real, torch.ones_like(pred_real))
            D_loss = (loss_real + loss_fake) * 0.5
            D_loss_epoch += D_loss
            D_loss.backward()
            D_optimizer.step()
            num_steps+=1
            itr+=1
            if num_steps%100 == 0:
                tqdm.write("Gen Loss: {:.6f}".format(D_loss))
                tqdm.write("Disc Loss: {:.6f}".format(G_loss))

                writer.add_scalars('Disc loss', {'D_loss':D_loss}, itr)
                writer.add_scalars('Gen loss', {'G_loss':G_loss}, itr)

        D_loss_epoch /= num_steps
        G_loss_epoch /= (num_steps * 3)

        tqdm.write("Steps p/ Epoch: {}".format(num_steps))
        # tqdm.write("Gen Loss: {:.6f}".format(D_loss_epoch))
        # tqdm.write("Disc Loss: {:.6f}".format(G_loss_epoch))

        # writer.add_scalars('losses', {'Disc loss': D_loss_epoch}, epoch)
        # writer.add_scalars('losses', {'Gen loss': G_loss_epoch}, epoch)
        state={
            'iter':itr,
            'epoch':epoch,
            'g_optim':G_optimizer.state_dict(),
            'd_optim':D_optimizer.state_dict(),
            'g_state_dict': generator.state_dict(),
            'd_state_dict': discriminator.state_dict(), 
        }
        torch.save(state, MODEL_PATH.format(model_name='{}'.format(experiment_name)))
        # torch.save(generator, MODEL_PATH.format(model_name='{}'.format(experiment_name+'_gen')))
        # torch.save(discriminator, MODEL_PATH.format(model_name='{}'.format(experiment_name+'_disc')))


        end_time = time.time()
        total_time = end_time - start_time
        medgan_val(target_validation_loader, source_validation_loader, generator, uNet, epoch, writer)
        tqdm.write("Epoch {} took {:.2f} seconds.".format(epoch, total_time))
        # if early_stopping:
        #     break


