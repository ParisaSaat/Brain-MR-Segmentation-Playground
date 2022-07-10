"""
Copyright (C) 2017 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
# from networks import AdaINGen, MsImageDis, VAEGen, UNet
from models.utils_codagan import weights_init, get_model_list, vgg_preprocess, get_scheduler, CrossEntropyLoss2d, norm, jaccard
from torch.autograd import Variable
import torch
import torch.nn as nn
# import torch.nn.functional as functional
import os
import wandb
import numpy as np

import torch.nn.functional as F
try:
    from itertools import izip as zip
except ImportError: # Will be 3.x series.
    pass

class MUNIT_Trainer(nn.Module):

    def __init__(self, hyperparameters, resume_epoch=-1, snapshot_dir=None):

        super(MUNIT_Trainer, self).__init__()

        lr = hyperparameters['lr']

        # Initiate the networks.
        self.gen = AdaINGen(hyperparameters['input_dim'] + hyperparameters['n_datasets'], hyperparameters['gen'], hyperparameters['n_datasets'])  # Auto-encoder for domain a.
        self.dis = MsImageDis(hyperparameters['input_dim'] + hyperparameters['n_datasets'], hyperparameters['dis'])  # Discriminator for domain a.

        self.instancenorm = nn.InstanceNorm2d(512, affine=False)
        self.style_dim = hyperparameters['gen']['style_dim']
        self.beta1 = hyperparameters['beta1']
        self.beta2 = hyperparameters['beta2']
        self.weight_decay = hyperparameters['weight_decay']

        # Initiating and loader pretrained UNet.
        self.sup = UNet(input_channels=hyperparameters['input_dim'], num_classes=2)

        # Fix the noise used in sampling.
        self.s_a = torch.randn(8, self.style_dim, 1, 1)
        self.s_b = torch.randn(8, self.style_dim, 1, 1)

        # Setup the optimizers.
        beta1 = hyperparameters['beta1']
        beta2 = hyperparameters['beta2']

        dis_params = list(self.dis.parameters())
        gen_params = list(self.gen.parameters()) + list(self.sup.parameters())

        self.dis_opt = torch.optim.Adam([p for p in dis_params if p.requires_grad],
                                        lr=lr, betas=(self.beta1, self.beta2), weight_decay=hyperparameters['weight_decay'])
        self.gen_opt = torch.optim.Adam([p for p in gen_params if p.requires_grad],
                                        lr=lr, betas=(self.beta1, self.beta2), weight_decay=hyperparameters['weight_decay'])

        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters)

        # Network weight initialization.
        self.apply(weights_init(hyperparameters['init']))
        self.dis.apply(weights_init('gaussian'))

        # Presetting one hot encoding vectors.
        self.one_hot_img = torch.zeros(hyperparameters['n_datasets'], hyperparameters['batch_size'], hyperparameters['n_datasets'], 256, 256)
        self.one_hot_c = torch.zeros(hyperparameters['n_datasets'], hyperparameters['batch_size'], hyperparameters['n_datasets'], 64, 64)

        for i in range(hyperparameters['n_datasets']):
            self.one_hot_img[i, :, i, :, :].fill_(1)
            self.one_hot_c[i, :, i, :, :].fill_(1)

        if resume_epoch == -1:
            self.wandb_id = wandb.util.generate_id()
            wandb.init(project="CODAGAN", entity="autoda", resume="allow", id=self.wandb_id,
                        config={})
        else:
            self.resume(snapshot_dir, hyperparameters)
        # if resume_epoch != -1:

        #     self.resume(snapshot_dir, hyperparameters)

    def recon_criterion(self, input, target):

        return torch.mean(torch.abs(input - target))

    def semi_criterion(self, input, target):

        loss = CrossEntropyLoss2d(size_average=False)
        return loss(input, target)

    def forward(self, x_a, x_b):

        self.eval()

        x_a.volatile = True
        x_b.volatile = True

        s_a = Variable(self.s_a, volatile=True)
        s_b = Variable(self.s_b, volatile=True)

        one_hot_x_a = torch.cat([x_a, self.one_hot_img[d_index_a]], 1)
        one_hot_x_b = torch.cat([x_b, self.one_hot_img[d_index_b]], 1)
        c_a, s_a_fake = self.gen.encode(one_hot_x_a)
        c_b, s_b_fake = self.gen.encode(one_hot_x_b)

        one_hot_c_b = torch.cat([c_b, self.one_hot_c[d_index_a]],  1)
        one_hot_c_a = torch.cat([c_a, self.one_hot_c[d_index_b]],  1)
        x_ba = self.gen.decode(one_hot_c_b, s_a)
        x_ab = self.gen.decode(one_hot_c_a, s_b)

        self.train()

        return x_ab, x_ba

    def set_gen_trainable(self, train_bool):

        if train_bool:
            self.gen.train()
            for param in self.gen.parameters():
                param.requires_grad = True

        else:
            self.gen.eval()
            for param in self.gen.parameters():
                param.requires_grad = True

    def set_sup_trainable(self, train_bool):

        if train_bool:
            self.sup.train()
            for param in self.sup.parameters():
                param.requires_grad = True
        else:
            self.sup.eval()
            for param in self.sup.parameters():
                param.requires_grad = True

    def sup_update(self, x_a, x_b, y_a, y_b, d_index_a, d_index_b, use_a, use_b, hyperparameters):

        self.gen_opt.zero_grad()

        s_a = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1))
        s_b = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1))

        one_hot_x_a = torch.cat([x_a, self.one_hot_img[d_index_a]], 1)
        one_hot_x_b = torch.cat([x_b, self.one_hot_img[d_index_b]], 1)

        # Encode.
        c_a, s_a_prime = self.gen.encode(one_hot_x_a)
        c_b, s_b_prime = self.gen.encode(one_hot_x_b)

        # Decode (within domain).
        one_hot_c_a = torch.cat([c_a, self.one_hot_c[d_index_a]],  1)
        one_hot_c_b = torch.cat([c_b, self.one_hot_c[d_index_b]],  1)
        x_a_recon = self.gen.decode(one_hot_c_a, s_a_prime)
        x_b_recon = self.gen.decode(one_hot_c_b, s_b_prime)

        # Decode (cross domain).
        one_hot_c_ab = torch.cat([c_a, self.one_hot_c[d_index_b]],  1)
        one_hot_c_ba = torch.cat([c_b, self.one_hot_c[d_index_a]],  1)
        x_ba = self.gen.decode(one_hot_c_ba, s_a)
        x_ab = self.gen.decode(one_hot_c_ab, s_b)

        # Encode again.
        one_hot_x_ba = torch.cat([x_ba, self.one_hot_img[d_index_a]], 1)
        one_hot_x_ab = torch.cat([x_ab, self.one_hot_img[d_index_b]], 1)
        c_b_recon, s_a_recon = self.gen.encode(one_hot_x_ba)
        c_a_recon, s_b_recon = self.gen.encode(one_hot_x_ab)

        # Forwarding through supervised model.
        p_a = None
        p_b = None
        loss_semi_a = None
        loss_semi_b = None

        # has_a_label = (c_a[use_a, :, :, :].size(0) != 0)
        has_a_label = (c_a[use_a.type(torch.bool), :, :, :].size(0) != 0)
        if has_a_label:
            p_a = self.sup(c_a, use_a, True)
            p_a_recon = self.sup(c_a_recon, use_a, True)
            # loss_semi_a = self.semi_criterion(p_a, y_a[use_a, :, :]) + \
            #               self.semi_criterion(p_a_recon, y_a[use_a, :, :])
            loss_semi_a = self.semi_criterion(p_a, y_a[use_a.type(torch.bool), :, :]) + \
                          self.semi_criterion(p_a_recon, y_a[use_a.type(torch.bool), :, :])

        # has_b_label = (c_b[use_b, :, :, :].size(0) != 0)
        has_b_label = (c_b[use_b.type(torch.bool), :, :, :].size(0) != 0)
        if has_b_label:
            p_b = self.sup(c_b, use_b, True)
            p_b_recon = self.sup(c_b, use_b, True)
            # loss_semi_b = self.semi_criterion(p_b, y_b[use_b, :, :]) + \
            #               self.semi_criterion(p_b_recon, y_b[use_b, :, :])
            loss_semi_b = self.semi_criterion(p_b, y_b[use_b.type(torch.bool), :, :]) + \
                          self.semi_criterion(p_b_recon, y_b[use_b.type(torch.bool), :, :])

        self.loss_gen_total = None
        if loss_semi_a is not None and loss_semi_b is not None:
            self.loss_gen_total = loss_semi_a + loss_semi_b
        elif loss_semi_a is not None:
            self.loss_gen_total = loss_semi_a
        elif loss_semi_b is not None:
            self.loss_gen_total = loss_semi_b

        if self.loss_gen_total is not None:
            self.loss_gen_total.backward()
            self.gen_opt.step()
            return self.loss_gen_total

    def sup_forward(self, x, y, d_index, hyperparameters):

        self.sup.eval()

        # Encoding content image.
        one_hot_x = torch.cat([x, self.one_hot_img[d_index, 0].unsqueeze(0)], 1)
        content, _ = self.gen.encode(one_hot_x)

        # Forwarding on supervised model.
        y_pred = self.sup(content, only_prediction=True)

        # Computing metrics.
        pred = y_pred.data.max(1)[1].squeeze_(1).squeeze_(0).cpu().numpy()

        jacc = jaccard(pred, y.cpu().squeeze(0).numpy())

        return jacc, pred, content

    def gen_update(self, x_a, x_b, d_index_a, d_index_b, hyperparameters):

        self.gen_opt.zero_grad()

        s_a = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1))
        s_b = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1))

        # Encode.
        one_hot_x_a = torch.cat([x_a, self.one_hot_img[d_index_a]], 1)
        one_hot_x_b = torch.cat([x_b, self.one_hot_img[d_index_b]], 1)

        c_a, s_a_prime = self.gen.encode(one_hot_x_a)
        c_b, s_b_prime = self.gen.encode(one_hot_x_b)

        # Decode (within domain).
        one_hot_c_a = torch.cat([c_a, self.one_hot_c[d_index_a]],  1)
        one_hot_c_b = torch.cat([c_b, self.one_hot_c[d_index_b]],  1)
        x_a_recon = self.gen.decode(one_hot_c_a, s_a_prime)
        x_b_recon = self.gen.decode(one_hot_c_b, s_b_prime)

        # Decode (cross domain).
        one_hot_c_ab = torch.cat([c_a, self.one_hot_c[d_index_b]],  1)
        one_hot_c_ba = torch.cat([c_b, self.one_hot_c[d_index_a]],  1)
        x_ba = self.gen.decode(one_hot_c_ba, s_a)
        x_ab = self.gen.decode(one_hot_c_ab, s_b)

        # Encode again.
        one_hot_x_ba = torch.cat([x_ba, self.one_hot_img[d_index_a]], 1)
        one_hot_x_ab = torch.cat([x_ab, self.one_hot_img[d_index_b]], 1)
        c_b_recon, s_a_recon = self.gen.encode(one_hot_x_ba)
        c_a_recon, s_b_recon = self.gen.encode(one_hot_x_ab)

        # Decode again (if needed).
        one_hot_c_aba_recon = torch.cat([c_a_recon, self.one_hot_c[d_index_a]],  1)
        one_hot_c_bab_recon = torch.cat([c_b_recon, self.one_hot_c[d_index_b]],  1)
        x_aba = self.gen.decode(one_hot_c_aba_recon, s_a_prime)
        x_bab = self.gen.decode(one_hot_c_bab_recon, s_b_prime)

        # Reconstruction loss.
        self.loss_gen_recon_x_a = self.recon_criterion(x_a_recon, x_a)
        self.loss_gen_recon_x_b = self.recon_criterion(x_b_recon, x_b)
        self.loss_gen_recon_s_a = self.recon_criterion(s_a_recon, s_a)
        self.loss_gen_recon_s_b = self.recon_criterion(s_b_recon, s_b)
        self.loss_gen_recon_c_a = self.recon_criterion(c_a_recon, c_a)
        self.loss_gen_recon_c_b = self.recon_criterion(c_b_recon, c_b)

        self.loss_gen_cycrecon_x_a = self.recon_criterion(x_aba, x_a)
        self.loss_gen_cycrecon_x_b = self.recon_criterion(x_bab, x_b)

        # GAN loss.
        self.loss_gen_adv_a = self.dis.calc_gen_loss(one_hot_x_ba)
        self.loss_gen_adv_b = self.dis.calc_gen_loss(one_hot_x_ab)

        # Total loss.
        self.loss_gen_total = hyperparameters['gan_w'] * self.loss_gen_adv_a + \
                              hyperparameters['gan_w'] * self.loss_gen_adv_b + \
                              hyperparameters['recon_x_w'] * self.loss_gen_recon_x_a + \
                              hyperparameters['recon_s_w'] * self.loss_gen_recon_s_a + \
                              hyperparameters['recon_c_w'] * self.loss_gen_recon_c_a + \
                              hyperparameters['recon_x_w'] * self.loss_gen_recon_x_b + \
                              hyperparameters['recon_s_w'] * self.loss_gen_recon_s_b + \
                              hyperparameters['recon_c_w'] * self.loss_gen_recon_c_b + \
                              hyperparameters['recon_x_cyc_w'] * self.loss_gen_cycrecon_x_a + \
                              hyperparameters['recon_x_cyc_w'] * self.loss_gen_cycrecon_x_b

        self.loss_gen_total.backward()
        self.gen_opt.step()
        return self.loss_gen_total

    def compute_vgg_loss(self, vgg, img, target):

        img_vgg = vgg_preprocess(img)
        target_vgg = vgg_preprocess(target)
        img_fea = vgg(img_vgg)
        target_fea = vgg(target_vgg)
        return torch.mean((self.instancenorm(img_fea) - self.instancenorm(target_fea)) ** 2)

    def sample(self, x_a, x_b):

        self.eval()
        x_a.volatile = True
        x_b.volatile = True
        s_a1 = Variable(self.s_a, volatile=True)
        s_b1 = Variable(self.s_b, volatile=True)
        s_a2 = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1), volatile=True)
        s_b2 = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1), volatile=True)
        x_a_recon, x_b_recon, x_ba1, x_ba2, x_ab1, x_ab2 = [], [], [], [], [], []
        for i in range(x_a.size(0)):

            one_hot_x_a = torch.cat([x_a[i].unsqueeze(0), self.one_hot_img_a[i].unsqueeze(0)], 1)
            one_hot_x_b = torch.cat([x_b[i].unsqueeze(0), self.one_hot_img_b[i].unsqueeze(0)], 1)

            c_a, s_a_fake = self.gen.encode(one_hot_x_a)
            c_b, s_b_fake = self.gen.encode(one_hot_x_b)
            x_a_recon.append(self.gen.decode(c_a, s_a_fake))
            x_b_recon.append(self.gen.decode(c_b, s_b_fake))
            x_ba1.append(self.gen.decode(c_b, s_a1[i].unsqueeze(0)))
            x_ba2.append(self.gen.decode(c_b, s_a2[i].unsqueeze(0)))
            x_ab1.append(self.gen.decode(c_a, s_b1[i].unsqueeze(0)))
            x_ab2.append(self.gen.decode(c_a, s_b2[i].unsqueeze(0)))

        x_a_recon, x_b_recon = torch.cat(x_a_recon), torch.cat(x_b_recon)
        x_ba1, x_ba2 = torch.cat(x_ba1), torch.cat(x_ba2)
        x_ab1, x_ab2 = torch.cat(x_ab1), torch.cat(x_ab2)
        self.train()
        return x_a, x_a_recon, x_ab1, x_ab2, x_b, x_b_recon, x_ba1, x_ba2

    def dis_update(self, x_a, x_b, d_index_a, d_index_b, hyperparameters):

        self.dis_opt.zero_grad()
        s_a = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1))
        s_b = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1))

        # Encode.
        one_hot_x_a = torch.cat([x_a, self.one_hot_img[d_index_a]], 1)
        one_hot_x_b = torch.cat([x_b, self.one_hot_img[d_index_b]], 1)

        c_a, _ = self.gen.encode(one_hot_x_a)
        c_b, _ = self.gen.encode(one_hot_x_b)

        one_hot_c_ba = torch.cat([c_b, self.one_hot_c[d_index_a]], 1)
        one_hot_c_ab = torch.cat([c_a, self.one_hot_c[d_index_b]], 1)

        # Decode (cross domain).
        x_ba = self.gen.decode(one_hot_c_ba, s_a)
        x_ab = self.gen.decode(one_hot_c_ab, s_b)

        # D loss.
        one_hot_x_ba = torch.cat([x_ba, self.one_hot_img[d_index_a]], 1)
        one_hot_x_ab = torch.cat([x_ab, self.one_hot_img[d_index_b]], 1)

        self.loss_dis_a = self.dis.calc_dis_loss(one_hot_x_ba, one_hot_x_a)
        self.loss_dis_b = self.dis.calc_dis_loss(one_hot_x_ab, one_hot_x_b)
        self.loss_dis_total = hyperparameters['gan_w'] * self.loss_dis_a + \
                              hyperparameters['gan_w'] * self.loss_dis_b
        self.loss_dis_total.backward()
        self.dis_opt.step()
        return self.loss_dis_total

    def update_learning_rate(self):

        if self.dis_scheduler is not None:
            self.dis_scheduler.step()
        if self.gen_scheduler is not None:
            self.gen_scheduler.step()

    def resume(self, checkpoint_dir, hyperparameters):

        print("--> " + checkpoint_dir)

        # Load generator.
        last_model_name = get_model_list(checkpoint_dir, "gen")
        state_dict = torch.load(last_model_name)
        self.gen.load_state_dict(state_dict)
        epochs = int(last_model_name[-11:-3])

        # Load supervised model.
        last_model_name = get_model_list(checkpoint_dir, "sup")
        state_dict = torch.load(last_model_name)
        self.sup.load_state_dict(state_dict)

        # Load discriminator.
        last_model_name = get_model_list(checkpoint_dir, "dis")
        state_dict = torch.load(last_model_name)
        self.dis.load_state_dict(state_dict)

        # Load optimizers.
        last_model_name = get_model_list(checkpoint_dir, "opt")
        state_dict = torch.load(last_model_name)
        self.dis_opt.load_state_dict(state_dict['dis'])
        self.gen_opt.load_state_dict(state_dict['gen'])

        for state in self.dis_opt.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v

        for state in self.gen_opt.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v

        # Reinitilize schedulers.
        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters, epochs)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters, epochs)

        print('Resume from epoch %d' % epochs)
        return epochs

    def save(self, snapshot_dir, epoch):

        # Save generators, discriminators, and optimizers.
        gen_name = os.path.join(snapshot_dir, 'gen_%08d.pt' % epoch)
        dis_name = os.path.join(snapshot_dir, 'dis_%08d.pt' % epoch)
        sup_name = os.path.join(snapshot_dir, 'sup_%08d.pt' % epoch)
        opt_name = os.path.join(snapshot_dir, 'opt_%08d.pt' % epoch)
        wandb_name = os.path.join(snapshot_dir, 'wandb_%08d.pt' % epoch)

        torch.save(self.gen.state_dict(), gen_name)
        torch.save(self.dis.state_dict(), dis_name)
        torch.save(self.sup.state_dict(), sup_name)
        torch.save({'gen': self.gen_opt.state_dict(), 'dis': self.dis_opt.state_dict()}, opt_name)
        torch.save({'wandb_id': self.wandb_id}, wandb_name)


class UNIT_Trainer(nn.Module):

    def __init__(self, hyperparameters, resume_epoch=-1, snapshot_dir=None):

        super(UNIT_Trainer, self).__init__()

        lr = hyperparameters['lr']

        # Initiate the networks.
        self.gen = VAEGen(hyperparameters['input_dim'] + hyperparameters['n_datasets'], hyperparameters['gen'], hyperparameters['n_datasets'])  # Auto-encoder for domain a.
        self.dis = MsImageDis(hyperparameters['input_dim'] + hyperparameters['n_datasets'], hyperparameters['dis'])  # Discriminator for domain a.

        self.instancenorm = nn.InstanceNorm2d(512, affine=False)

        self.sup = UNet(input_channels=hyperparameters['input_dim'], num_classes=2)

        # Setup the optimizers.
        beta1 = hyperparameters['beta1']
        beta2 = hyperparameters['beta2']
        dis_params = list(self.dis.parameters())
        gen_params = list(self.gen.parameters()) + list(self.sup.parameters())
        self.dis_opt = torch.optim.Adam([p for p in dis_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.gen_opt = torch.optim.Adam([p for p in gen_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters)

        # Network weight initialization.
        self.apply(weights_init(hyperparameters['init']))
        self.dis.apply(weights_init('gaussian'))

        # Presetting one hot encoding vectors.
        self.one_hot_img = torch.zeros(hyperparameters['n_datasets'], hyperparameters['batch_size'], hyperparameters['n_datasets'], 256, 256)
        self.one_hot_h = torch.zeros(hyperparameters['n_datasets'], hyperparameters['batch_size'], hyperparameters['n_datasets'], 64, 64)

        for i in range(hyperparameters['n_datasets']):
            self.one_hot_img[i, :, i, :, :].fill_(1)
            self.one_hot_h[i, :, i, :, :].fill_(1)


        if resume_epoch != -1:

            self.resume(snapshot_dir, hyperparameters)


    def recon_criterion(self, input, target):

        return torch.mean(torch.abs(input - target))

    def semi_criterion(self, input, target):

        loss = CrossEntropyLoss2d(size_average=False)
        return loss(input, target)

    def forward(self, x_a, x_b):

        self.eval()
        x_a.volatile = True
        x_b.volatile = True
        h_a, _ = self.gen_a.encode(x_a)
        h_b, _ = self.gen_b.encode(x_b)
        x_ba = self.gen_a.decode(h_b)
        x_ab = self.gen_b.decode(h_a)
        self.train()
        return x_ab, x_ba

    def __compute_kl(self, mu):

        # def _compute_kl(self, mu, sd):
        # mu_2 = torch.pow(mu, 2)
        # sd_2 = torch.pow(sd, 2)
        # encoding_loss = (mu_2 + sd_2 - torch.log(sd_2)).sum() / mu_2.size(0)
        # return encoding_loss

        mu_2 = torch.pow(mu, 2)
        encoding_loss = torch.mean(mu_2)
        return encoding_loss

    def set_gen_trainable(self, train_bool):

        if train_bool:
            self.gen.train()
            for param in self.gen.parameters():
                param.requires_grad = True

        else:
            self.gen.eval()
            for param in self.gen.parameters():
                param.requires_grad = True

    def set_sup_trainable(self, train_bool):

        if train_bool:
            self.sup.train()
            for param in self.sup.parameters():
                param.requires_grad = True
        else:
            self.sup.eval()
            for param in self.sup.parameters():
                param.requires_grad = True

    def sup_update(self, x_a, x_b, y_a, y_b, d_index_a, d_index_b, use_a, use_b, hyperparameters):

        self.gen_opt.zero_grad()

        # Encode.
        one_hot_x_a = torch.cat([x_a, self.one_hot_img[d_index_a]], 1)
        one_hot_x_b = torch.cat([x_b, self.one_hot_img[d_index_b]], 1)
        h_a, n_a = self.gen.encode(one_hot_x_a)
        h_b, n_b = self.gen.encode(one_hot_x_b)

        # Decode (within domain).
        one_hot_h_a = torch.cat([h_a + n_a, self.one_hot_h[d_index_a]], 1)
        one_hot_h_b = torch.cat([h_b + n_b, self.one_hot_h[d_index_b]], 1)
        x_a_recon = self.gen.decode(one_hot_h_a)
        x_b_recon = self.gen.decode(one_hot_h_b)

        # Decode (cross domain).
        one_hot_h_ab = torch.cat([h_a + n_a, self.one_hot_h[d_index_b]], 1)
        one_hot_h_ba = torch.cat([h_b + n_b, self.one_hot_h[d_index_a]], 1)
        x_ba = self.gen.decode(one_hot_h_ba)
        x_ab = self.gen.decode(one_hot_h_ab)

        # Encode again.
        one_hot_x_ba = torch.cat([x_ba, self.one_hot_img[d_index_a]], 1)
        one_hot_x_ab = torch.cat([x_ab, self.one_hot_img[d_index_b]], 1)
        h_b_recon, n_b_recon = self.gen.encode(one_hot_x_ba)
        h_a_recon, n_a_recon = self.gen.encode(one_hot_x_ab)

        # Decode again (if needed).
        one_hot_h_a_recon = torch.cat([h_a_recon + n_a_recon, self.one_hot_h[d_index_a]], 1)
        one_hot_h_b_recon = torch.cat([h_b_recon + n_b_recon, self.one_hot_h[d_index_b]], 1)
        x_aba = self.gen.decode(one_hot_h_a_recon) if hyperparameters['recon_x_cyc_w'] > 0 else None
        x_bab = self.gen.decode(one_hot_h_b_recon) if hyperparameters['recon_x_cyc_w'] > 0 else None

        # Forwarding through supervised model.
        p_a = None
        p_b = None
        loss_semi_a = None
        loss_semi_b = None

        has_a_label = (h_a[use_a, :, :, :].size(0) != 0)
        if has_a_label:
            p_a = self.sup(h_a, use_a, True)
            p_a_recon = self.sup(h_a_recon, use_a, True)
            loss_semi_a = self.semi_criterion(p_a, y_a[use_a, :, :]) + \
                          self.semi_criterion(p_a_recon, y_a[use_a, :, :])

        has_b_label = (h_b[use_b, :, :, :].size(0) != 0)
        if has_b_label:
            p_b = self.sup(h_b, use_b, True)
            p_b_recon = self.sup(h_b, use_b, True)
            loss_semi_b = self.semi_criterion(p_b, y_b[use_b, :, :]) + \
                          self.semi_criterion(p_b_recon, y_b[use_b, :, :])

        self.loss_gen_total = None
        if loss_semi_a is not None and loss_semi_b is not None:
            self.loss_gen_total = loss_semi_a + loss_semi_b
        elif loss_semi_a is not None:
            self.loss_gen_total = loss_semi_a
        elif loss_semi_b is not None:
            self.loss_gen_total = loss_semi_b

        if self.loss_gen_total is not None:
            self.loss_gen_total.backward()
            self.gen_opt.step()

    def sup_forward(self, x, y, d_index, hyperparameters):

        self.sup.eval()

        # Encoding content image.
        one_hot_x = torch.cat([x, self.one_hot_img[d_index, 0].unsqueeze(0)], 1)
        hidden, _ = self.gen.encode(one_hot_x)

        # Forwarding on supervised model.
        y_pred = self.sup(hidden, only_prediction=True)

        # Computing metrics.
        pred = y_pred.data.max(1)[1].squeeze_(1).squeeze_(0).cpu().numpy()

        jacc = jaccard(pred, y.cpu().squeeze(0).numpy())

        return jacc, pred, hidden

    def gen_update(self, x_a, x_b, d_index_a, d_index_b, hyperparameters):

        self.gen_opt.zero_grad()

        # Encode.
        one_hot_x_a = torch.cat([x_a, self.one_hot_img[d_index_a]], 1)
        one_hot_x_b = torch.cat([x_b, self.one_hot_img[d_index_b]], 1)
        h_a, n_a = self.gen.encode(one_hot_x_a)
        h_b, n_b = self.gen.encode(one_hot_x_b)

        # Decode (within domain).
        one_hot_h_a = torch.cat([h_a + n_a, self.one_hot_h[d_index_a]], 1)
        one_hot_h_b = torch.cat([h_b + n_b, self.one_hot_h[d_index_b]], 1)
        x_a_recon = self.gen.decode(one_hot_h_a)
        x_b_recon = self.gen.decode(one_hot_h_b)

        # Decode (cross domain).
        one_hot_h_ab = torch.cat([h_a + n_a, self.one_hot_h[d_index_b]], 1)
        one_hot_h_ba = torch.cat([h_b + n_b, self.one_hot_h[d_index_a]], 1)
        x_ba = self.gen.decode(one_hot_h_ba)
        x_ab = self.gen.decode(one_hot_h_ab)

        # Encode again.
        one_hot_x_ba = torch.cat([x_ba, self.one_hot_img[d_index_a]], 1)
        one_hot_x_ab = torch.cat([x_ab, self.one_hot_img[d_index_b]], 1)
        h_b_recon, n_b_recon = self.gen.encode(one_hot_x_ba)
        h_a_recon, n_a_recon = self.gen.encode(one_hot_x_ab)

        # Decode again (if needed).
        one_hot_h_a_recon = torch.cat([h_a_recon + n_a_recon, self.one_hot_h[d_index_a]], 1)
        one_hot_h_b_recon = torch.cat([h_b_recon + n_b_recon, self.one_hot_h[d_index_b]], 1)
        x_aba = self.gen.decode(one_hot_h_a_recon) if hyperparameters['recon_x_cyc_w'] > 0 else None
        x_bab = self.gen.decode(one_hot_h_b_recon) if hyperparameters['recon_x_cyc_w'] > 0 else None

        # Reconstruction loss.
        self.loss_gen_recon_x_a = self.recon_criterion(x_a_recon, x_a)
        self.loss_gen_recon_x_b = self.recon_criterion(x_b_recon, x_b)
        self.loss_gen_recon_kl_a = self.__compute_kl(h_a)
        self.loss_gen_recon_kl_b = self.__compute_kl(h_b)
        self.loss_gen_cyc_x_a = self.recon_criterion(x_aba, x_a)
        self.loss_gen_cyc_x_b = self.recon_criterion(x_bab, x_b)
        self.loss_gen_recon_kl_cyc_aba = self.__compute_kl(h_a_recon)
        self.loss_gen_recon_kl_cyc_bab = self.__compute_kl(h_b_recon)

        # GAN loss.
        self.loss_gen_adv_a = self.dis.calc_gen_loss(one_hot_x_ba)
        self.loss_gen_adv_b = self.dis.calc_gen_loss(one_hot_x_ab)

        # Total loss.
        self.loss_gen_total = hyperparameters['gan_w'] * self.loss_gen_adv_a + \
                              hyperparameters['gan_w'] * self.loss_gen_adv_b + \
                              hyperparameters['recon_x_w'] * self.loss_gen_recon_x_a + \
                              hyperparameters['recon_kl_w'] * self.loss_gen_recon_kl_a + \
                              hyperparameters['recon_x_w'] * self.loss_gen_recon_x_b + \
                              hyperparameters['recon_kl_w'] * self.loss_gen_recon_kl_b + \
                              hyperparameters['recon_x_cyc_w'] * self.loss_gen_cyc_x_a + \
                              hyperparameters['recon_kl_cyc_w'] * self.loss_gen_recon_kl_cyc_aba + \
                              hyperparameters['recon_x_cyc_w'] * self.loss_gen_cyc_x_b + \
                              hyperparameters['recon_kl_cyc_w'] * self.loss_gen_recon_kl_cyc_bab

        self.loss_gen_total.backward()
        self.gen_opt.step()

    def sample(self, x_a, x_b):

        self.eval()
        x_a.volatile = True
        x_b.volatile = True
        x_a_recon, x_b_recon, x_ba, x_ab = [], [], [], []
        for i in range(x_a.size(0)):
            h_a, _ = self.gen_a.encode(x_a[i].unsqueeze(0))
            h_b, _ = self.gen_b.encode(x_b[i].unsqueeze(0))
            x_a_recon.append(self.gen_a.decode(h_a))
            x_b_recon.append(self.gen_b.decode(h_b))
            x_ba.append(self.gen_a.decode(h_b))
            x_ab.append(self.gen_b.decode(h_a))
        x_a_recon, x_b_recon = torch.cat(x_a_recon), torch.cat(x_b_recon)
        x_ba = torch.cat(x_ba)
        x_ab = torch.cat(x_ab)
        self.train()
        return x_a, x_a_recon, x_ab, x_b, x_b_recon, x_ba

    def dis_update(self, x_a, x_b, d_index_a, d_index_b, hyperparameters):

        self.dis_opt.zero_grad()

        # Encode.
        one_hot_x_a = torch.cat([x_a, self.one_hot_img[d_index_a]], 1)
        one_hot_x_b = torch.cat([x_b, self.one_hot_img[d_index_b]], 1)
        h_a, n_a = self.gen.encode(one_hot_x_a)
        h_b, n_b = self.gen.encode(one_hot_x_b)

        # Decode (cross domain).
        one_hot_h_ab = torch.cat([h_a + n_a, self.one_hot_h[d_index_b]], 1)
        one_hot_h_ba = torch.cat([h_b + n_b, self.one_hot_h[d_index_a]], 1)
        x_ba = self.gen.decode(one_hot_h_ba)
        x_ab = self.gen.decode(one_hot_h_ab)

        # D loss.
        one_hot_x_ba = torch.cat([x_ba, self.one_hot_img[d_index_a]], 1)
        one_hot_x_ab = torch.cat([x_ab, self.one_hot_img[d_index_b]], 1)
        self.loss_dis_a = self.dis.calc_dis_loss(one_hot_x_ba.detach(), one_hot_x_a)
        self.loss_dis_b = self.dis.calc_dis_loss(one_hot_x_ab.detach(), one_hot_x_b)
        self.loss_dis_total = hyperparameters['gan_w'] * self.loss_dis_a + \
                              hyperparameters['gan_w'] * self.loss_dis_b

        self.loss_dis_total.backward()
        self.dis_opt.step()

    def update_learning_rate(self):

        if self.dis_scheduler is not None:
            self.dis_scheduler.step()
        if self.gen_scheduler is not None:
            self.gen_scheduler.step()

    def resume(self, checkpoint_dir, hyperparameters):

        self.wandb_id = checkpoint_dir["wandb_id"]
        wandb.init(project="CODAGAN", entity="autoda", resume="allow", id=self.wandb_id,
                        config={})
        # Load generators.
        last_model_name = get_model_list(checkpoint_dir, "gen")
        state_dict = torch.load(last_model_name)
        self.gen.load_state_dict(state_dict)
        epochs = int(last_model_name[-11:-3])

        # Load discriminators.
        last_model_name = get_model_list(checkpoint_dir, "dis")
        state_dict = torch.load(last_model_name)
        self.dis.load_state_dict(state_dict)

        # Load supervised model.
        last_model_name = get_model_list(checkpoint_dir, "sup")
        state_dict = torch.load(last_model_name)
        self.sup.load_state_dict(state_dict)

        # Load optimizers.
        last_model_name = get_model_list(checkpoint_dir, "opt")
        state_dict = torch.load(last_model_name)
        self.dis_opt.load_state_dict(state_dict['dis'])
        self.gen_opt.load_state_dict(state_dict['gen'])

        for state in self.dis_opt.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v

        for state in self.gen_opt.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v

        # Reinitilize schedulers.
        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters, epochs)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters, epochs)

        print('Resume from iteration %d' % epochs)
        return epochs

    def save(self, snapshot_dir, epoch):

        # Save generators, discriminators, and optimizers.
        gen_name = os.path.join(snapshot_dir, 'gen_%08d.pt' % epoch)
        dis_name = os.path.join(snapshot_dir, 'dis_%08d.pt' % epoch)
        sup_name = os.path.join(snapshot_dir, 'sup_%08d.pt' % epoch)
        opt_name = os.path.join(snapshot_dir, 'opt_%08d.pt' % epoch)

        torch.save(self.gen.state_dict(), gen_name)
        torch.save(self.dis.state_dict(), dis_name)
        torch.save(self.sup.state_dict(), sup_name)
        torch.save({'dis': self.dis_opt.state_dict(), 'gen': self.gen_opt.state_dict()}, opt_name)


##################################################################################
# Discriminator
##################################################################################

class MsImageDis(nn.Module):

    # Multi-scale discriminator architecture.
    def __init__(self, input_dim, params):

        super(MsImageDis, self).__init__()

        self.n_layer = params['n_layer']
        self.gan_type = params['gan_type']
        self.dim = params['dim']
        self.norm = params['norm']
        self.activ = params['activ']
        self.num_scales = params['num_scales']
        self.pad_type = params['pad_type']
        self.input_dim = input_dim
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
        self.cnns = nn.ModuleList()

        for _ in range(self.num_scales):
            self.cnns.append(self._make_net())

    def _make_net(self):

        dim = self.dim
        cnn_x = []
        cnn_x += [Conv2dBlock(self.input_dim, dim, 4, 2, 1, norm='none', activation=self.activ, pad_type=self.pad_type)]
        for i in range(self.n_layer - 1):
            cnn_x += [Conv2dBlock(dim, dim * 2, 4, 2, 1, norm=self.norm, activation=self.activ, pad_type=self.pad_type)]
            dim *= 2
        cnn_x += [nn.Conv2d(dim, 1, 1, 1, 0)]
        cnn_x = nn.Sequential(*cnn_x)
        return cnn_x

    def forward(self, x):

        outputs = []
        for model in self.cnns:
            outputs.append(model(x))
            x = self.downsample(x)
        return outputs

    def calc_dis_loss(self, input_fake, input_real):

        # Calculate the loss to train D.
        outs0 = self.forward(input_fake)
        outs1 = self.forward(input_real)
        loss = 0

        for it, (out0, out1) in enumerate(zip(outs0, outs1)):
            if self.gan_type == 'lsgan':
                loss += torch.mean((out0 - 0)**2) + torch.mean((out1 - 1)**2)
            elif self.gan_type == 'nsgan':
                all0 = Variable(torch.zeros_like(out0.data), requires_grad=False)
                all1 = Variable(torch.ones_like(out1.data), requires_grad=False)
                loss += torch.mean(F.binary_cross_entropy(F.sigmoid(out0), all0) +
                                   F.binary_cross_entropy(F.sigmoid(out1), all1))
            else:
                assert 0, "Unsupported GAN type: {}".format(self.gan_type)
        return loss

    def calc_gen_loss(self, input_fake):
        # Calculate the loss to train G.
        outs0 = self.forward(input_fake)
        loss = 0
        for it, (out0) in enumerate(outs0):
            if self.gan_type == 'lsgan':
                loss += torch.mean((out0 - 1)**2) # LSGAN.
            elif self.gan_type == 'nsgan':
                all1 = Variable(torch.ones_like(out0.data), requires_grad=False)
                loss += torch.mean(F.binary_cross_entropy(F.sigmoid(out0), all1))
            else:
                assert 0, "Unsupported GAN type: {}".format(self.gan_type)
        return loss

##################################################################################
# Generator
##################################################################################

class AdaINGen(nn.Module):

    # AdaIN auto-encoder architecture.
    def __init__(self, input_dim, params, n_datasets):

        super(AdaINGen, self).__init__()

        dim = params['dim']
        style_dim = params['style_dim']
        n_downsample = params['n_downsample']
        n_res = params['n_res']
        activ = params['activ']
        pad_type = params['pad_type']
        mlp_dim = params['mlp_dim']

        # Style encoder.
        self.enc_style = StyleEncoder(4, input_dim, dim, style_dim, norm='none', activ=activ, pad_type=pad_type)

        # Content encoder.
        self.enc_content = ContentEncoder(n_downsample, n_res, input_dim, dim, 'in', activ, pad_type=pad_type)
        self.dec = Decoder(n_downsample, n_res, self.enc_content.output_dim + n_datasets, input_dim - n_datasets, res_norm='adain', activ=activ, pad_type=pad_type)

        # MLP to generate AdaIN parameters.
        self.mlp = MLP(style_dim, self.get_num_adain_params(self.dec), mlp_dim, 3, norm='none', activ=activ)

    def forward(self, images):

        # Reconstruct an image.
        content, style_fake = self.encode(images)
        images_recon = self.decode(content, style_fake)
        return images_recon

    def encode(self, images):

        # Encode an image to its content and style codes.
        style_fake = self.enc_style(images)
        content = self.enc_content(images)
        return content, style_fake

    def decode(self, content, style):

        # Decode content and style codes to an image.
        adain_params = self.mlp(style)
        self.assign_adain_params(adain_params, self.dec)
        images = self.dec(content)
        return images

    def assign_adain_params(self, adain_params, model):

        # Assign the adain_params to the AdaIN layers in model.
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                mean = adain_params[:, :m.num_features]
                std = adain_params[:, m.num_features:2*m.num_features]
                m.bias = mean.contiguous().view(-1)
                m.weight = std.contiguous().view(-1)
                if adain_params.size(1) > 2*m.num_features:
                    adain_params = adain_params[:, 2*m.num_features:]

    def get_num_adain_params(self, model):

        # Return the number of AdaIN parameters needed by the model.
        num_adain_params = 0
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                num_adain_params += 2*m.num_features
        return num_adain_params


class VAEGen(nn.Module):

    # VAE architecture.
    def __init__(self, input_dim, params, n_datasets):

        super(VAEGen, self).__init__()

        dim = params['dim']
        n_downsample = params['n_downsample']
        n_res = params['n_res']
        activ = params['activ']
        pad_type = params['pad_type']

        # Content encoder.
        self.enc = ContentEncoder(n_downsample, n_res, input_dim, dim, 'in', activ, pad_type=pad_type)
        self.dec = Decoder(n_downsample, n_res, self.enc.output_dim + n_datasets, input_dim - n_datasets, res_norm='in', activ=activ, pad_type=pad_type)

    def forward(self, images):

        # This is a reduced VAE implementation where we assume the outputs are multivariate Gaussian distribution with mean = hiddens and std_dev = all ones.
        hiddens = self.encode(images)
        if self.training == True:
            noise = Variable(torch.randn(hiddens.size()).cuda(hiddens.data.get_device()))
            images_recon = self.decode(hiddens + noise)
        else:
            images_recon = self.decode(hiddens)
        return images_recon, hiddens

    def encode(self, images):

        hiddens = self.enc(images)
        noise = Variable(torch.randn(hiddens.size()).cuda(hiddens.data.get_device()))
        return hiddens, noise

    def decode(self, hiddens):

        images = self.dec(hiddens)
        return images


##################################################################################
# Encoder and Decoders
##################################################################################

class StyleEncoder(nn.Module):

    def __init__(self, n_downsample, input_dim, dim, style_dim, norm, activ, pad_type):

        super(StyleEncoder, self).__init__()

        self.model = []
        self.model += [Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)]
        for i in range(2):
            self.model += [Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
            dim *= 2
        for i in range(n_downsample - 2):
            self.model += [Conv2dBlock(dim, dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
        self.model += [nn.AdaptiveAvgPool2d(1)] # Global average pooling.
        self.model += [nn.Conv2d(dim, style_dim, 1, 1, 0)]
        self.model = nn.Sequential(*self.model)
        self.output_dim = dim

    def forward(self, x):

        return self.model(x)

class ContentEncoder(nn.Module):

    def __init__(self, n_downsample, n_res, input_dim, dim, norm, activ, pad_type):

        super(ContentEncoder, self).__init__()

        self.model = []
        self.model += [Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)]

        # Downsampling blocks.
        for i in range(n_downsample):
            self.model += [Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
            dim *= 2
        # Residual blocks.
        self.model += [ResBlocks(n_res, dim, norm=norm, activation=activ, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)
        self.output_dim = dim

    def forward(self, x):

        return self.model(x)

class Decoder(nn.Module):

    def __init__(self, n_upsample, n_res, dim, output_dim, res_norm='adain', activ='relu', pad_type='zero'):

        super(Decoder, self).__init__()

        self.model = []

        # AdaIN residual blocks.
        self.model += [ResBlocks(n_res, dim, res_norm, activ, pad_type=pad_type)]

        # Upsampling blocks.
        for i in range(n_upsample):
            self.model += [nn.Upsample(scale_factor=2),
                           Conv2dBlock(dim, dim // 2, 5, 1, 2, norm='ln', activation=activ, pad_type=pad_type)]
            dim //= 2

        # Use reflection padding in the last conv layer.
        self.model += [Conv2dBlock(dim, output_dim, 7, 1, 3, norm='none', activation='tanh', pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):

        return self.model(x)

##################################################################################
# Sequential Models
##################################################################################
class ResBlocks(nn.Module):

    def __init__(self, num_blocks, dim, norm='in', activation='relu', pad_type='zero'):

        super(ResBlocks, self).__init__()

        self.model = []
        for i in range(num_blocks):
            self.model += [ResBlock(dim, norm=norm, activation=activation, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):

        return self.model(x)

class MLP(nn.Module):

    def __init__(self, input_dim, output_dim, dim, n_blk, norm='none', activ='relu'):

        super(MLP, self).__init__()

        self.model = []
        self.model += [LinearBlock(input_dim, dim, norm=norm, activation=activ)]
        for i in range(n_blk - 2):
            self.model += [LinearBlock(dim, dim, norm=norm, activation=activ)]
        self.model += [LinearBlock(dim, output_dim, norm='none', activation='none')] # no output activations
        self.model = nn.Sequential(*self.model)

    def forward(self, x):

        return self.model(x.view(x.size(0), -1))

##################################################################################
# Basic Blocks
##################################################################################
class ResBlock(nn.Module):

    def __init__(self, dim, norm='in', activation='relu', pad_type='zero'):

        super(ResBlock, self).__init__()

        model = []
        model += [Conv2dBlock(dim, dim, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type)]
        model += [Conv2dBlock(dim, dim, 3, 1, 1, norm=norm, activation='none', pad_type=pad_type)]
        self.model = nn.Sequential(*model)

    def forward(self, x):

        residual = x
        out = self.model(x)
        out += residual
        return out

class Conv2dBlock(nn.Module):

    def __init__(self, input_dim ,output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', pad_type='zero'):

        super(Conv2dBlock, self).__init__()

        self.use_bias = True

        # Initialize padding.
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # Initialize normalization.
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNorm2d(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # Initialize activation.
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # Initialize convolution.
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias)

    def forward(self, x):

        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x

class LinearBlock(nn.Module):

    def __init__(self, input_dim, output_dim, norm='none', activation='relu'):

        super(LinearBlock, self).__init__()

        use_bias = True

        # Initialize fully connected layer.
        self.fc = nn.Linear(input_dim, output_dim, bias=use_bias)

        # Initialize normalization.
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm1d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm1d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # Initialize activation.
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

    def forward(self, x):

        out = self.fc(x)
        if self.norm:
            out = self.norm(out)
        if self.activation:
            out = self.activation(out)
        return out

##################################################################################
# VGG network definition
##################################################################################
class Vgg16(nn.Module):

    def __init__(self):

        super(Vgg16, self).__init__()

        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

    def forward(self, X):

        h = F.relu(self.conv1_1(X), inplace=True)
        h = F.relu(self.conv1_2(h), inplace=True)

        # relu1_2 = h.
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv2_1(h), inplace=True)
        h = F.relu(self.conv2_2(h), inplace=True)

        # relu2_2 = h.
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv3_1(h), inplace=True)
        h = F.relu(self.conv3_2(h), inplace=True)
        h = F.relu(self.conv3_3(h), inplace=True)

        # relu3_3 = h.
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv4_1(h), inplace=True)
        h = F.relu(self.conv4_2(h), inplace=True)
        h = F.relu(self.conv4_3(h), inplace=True)

        # relu4_3 = h.
        h = F.relu(self.conv5_1(h), inplace=True)
        h = F.relu(self.conv5_2(h), inplace=True)
        h = F.relu(self.conv5_3(h), inplace=True)
        relu5_3 = h

        return relu5_3
        # return [relu1_2, relu2_2, relu3_3, relu4_3]

##################################################################################
# Normalization layers
##################################################################################
class AdaptiveInstanceNorm2d(nn.Module):

    def __init__(self, num_features, eps=1e-5, momentum=0.1):

        super(AdaptiveInstanceNorm2d, self).__init__()

        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        # Weight and bias are dynamically assigned.
        self.weight = None
        self.bias = None

        # Just dummy buffers, not used.
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):

        assert self.weight is not None and self.bias is not None, "Please assign weight and bias before calling AdaIN!"
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)

        # Apply instance norm.
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])

        out = F.batch_norm(
            x_reshaped, running_mean, running_var, self.weight, self.bias,
            True, self.momentum, self.eps)

        return out.view(b, c, *x.size()[2:])

    def __repr__(self):

        return self.__class__.__name__ + '(' + str(self.num_features) + ')'

class LayerNorm(nn.Module):

    def __init__(self, num_features, eps=1e-5, affine=True):

        super(LayerNorm, self).__init__()

        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):

        shape = [-1] + [1] * (x.dim() - 1)
        mean = x.view(x.size(0), -1).mean(1).view(*shape)
        std = x.view(x.size(0), -1).std(1).view(*shape)
        x = (x - mean) / (std + self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)

        return x


####################################################################################
# UNet Architecture. ###############################################################
# Adapted from: https://github.com/zijundeng/pytorch-semantic-segmentation/ ########
####################################################################################

class _EncoderBlock(nn.Module):

    def __init__(self, in_channels, out_channels, dropout=False):

        super(_EncoderBlock, self).__init__()

        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]

        if dropout:

            layers.append(nn.Dropout())

        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        self.encode = nn.Sequential(*layers)

    def forward(self, x):

        return self.encode(x)


class _DecoderBlock(nn.Module):

    def __init__(self, in_channels, middle_channels, out_channels):

        super(_DecoderBlock, self).__init__()

        self.decode = nn.Sequential(
            nn.Dropout2d(),
            nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=2, stride=2, padding=0, output_padding=0)
        )

    def forward(self, x):

        return self.decode(x)

class UNet(nn.Module):

    def __init__(self, input_channels, num_classes):

        super(UNet, self).__init__()

        self.drop = nn.Dropout2d(p=0.5)

        # Encoders.
        # First two layers not used due to asymmetric
        # Encoder-Decoder architectures due to UNIT/MUNIT downsampling.
        #self.enc1 = _EncoderBlock(input_channels, 64)
        #self.enc2 = _EncoderBlock(64, 128)
        self.enc3 = _EncoderBlock(128, 256)
        self.enc4 = _EncoderBlock(256, 512, dropout=True)


        # Decoders.
        self.center = _DecoderBlock(512, 1024, 512)

        self.dec4 = _DecoderBlock(1024, 512, 256)
        self.dec3 = _DecoderBlock(512, 256, 128)
        self.dec2 = _DecoderBlock(128, 64, 64)

        self.dec1 = nn.Sequential(
            nn.Dropout2d(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Class prediction.
        self.final = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x, use=None, only_prediction=True):

        x = self.drop(x)

        # Encoding.
        enc3 = self.enc3(x)    # 64 -> 32.
        enc4 = self.enc4(enc3) # 32 -> 16.

        # Decoding.
        center = self.center(enc4) # 16 -> 32.

        # dec4 = self.dec4(torch.cat([center, F.upsample(enc4, center.size()[2:], mode='bilinear')], 1)) # 32 -> 64.
        # dec3 = self.dec3(torch.cat([dec4, F.upsample(enc3, dec4.size()[2:], mode='bilinear')], 1)) # 64 -> 128.
        dec4 = self.dec4(torch.cat([center, nn.functional.interpolate(enc4, center.size()[2:], mode='bilinear')], 1)) # 32 -> 64.
        dec3 = self.dec3(torch.cat([dec4, nn.functional.interpolate(enc3, dec4.size()[2:], mode='bilinear')], 1)) # 64 -> 128.
        dec2 = self.dec2(dec3) # 128 -> 256.
        dec1 = self.dec1(dec2) # 256 -> 256.

        final = self.final(dec1) # 256 -> 256.

        # Returning tensors.
        if only_prediction:
            if use is not None:
                # return final[use,:,:,:]
                return final[use.type(torch.bool),:,:,:]
            else:
                return final
        else:

            if use is not None:
                return final[use,:,:,:], [dec4, dec3, dec2, dec1]
            else:
                return final, [dec4, dec3, dec2, dec1]
