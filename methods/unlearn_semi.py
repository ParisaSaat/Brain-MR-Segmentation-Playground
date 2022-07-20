import os.path
import sys

import numpy as np
import torch
import wandb
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from torch.autograd import Variable

from config.io import *
#from data.utils import get_dataloader
from metrics.confusion_loss import confusion_loss
from metrics.dice import dice_loss
from models.unlearn import UNet, Segmenter, DomainPredictor
from models.utils import EarlyStoppingUnlearning
from data_data.utils import get_dataloader

def train_encoder_domain_unlearn_semi(args, models, train_loaders, optimizers, criterions, epoch, problem):
    cuda = torch.cuda.is_available()

    [encoder, regressor, domain_predictor] = models
    [optimizer] = optimizers
    [source_train_dataloader, target_train_dataloader, _, _] = train_loaders
    [criteron, _, domain_criterion] = criterions
    regressor_loss = 0
    domain_loss = 0

    encoder.train()
    regressor.train()
    domain_predictor.train()

    true_domains = []
    pred_domains = []

    batches = 0
    for batch_idx, (source_load, target_load) in enumerate(zip(source_train_dataloader, target_train_dataloader)):
        s_data, s_target, s_domain = source_load['image'], source_load['mask'], source_load['domain']
        t_data, t_target, t_domain = target_load['image'], target_load['mask'], target_load['domain']
        n1 = np.random.randint(1, len(s_data) - 1)
        n2 = len(s_data) - n1

        s_data = s_data[:n1]
        s_target = s_target[:n1]
        s_domain = s_domain[:n1]

        t_data = t_data[:n2]
        t_target = t_target[:n2]
        t_domain = t_domain[:n2]

        data = torch.cat((s_data, t_data), 0)
        target = torch.cat((s_target, t_target), 0)
        domain_target = torch.cat((s_domain, t_domain), 0)
        target = target.type(torch.LongTensor)

        if cuda:
            data, target, domain_target = data.cuda(), target.cuda(), domain_target.cuda()

        data, target, domain_target = Variable(data), Variable(target), Variable(domain_target)

        if list(data.size())[0] == args["batch_size"]:

            batches += 1

            # First update the encoder and regressor for now dont improve the domain stuff, just the feature predictions
            optimizer.zero_grad()
            features = encoder(data.unsqueeze(1))
            output_pred = regressor(features)

            op_0 = output_pred[:n1]
            target_0 = target[:n1]

            op_1 = output_pred[n1:]
            target_1 = target[n1:]

            if problem == 'wgc':
                one_hot_mask_0 = torch.nn.functional.one_hot(target_0.long(), num_classes=4).transpose(1, -1).squeeze(
                    -1)
                target_0 = one_hot_mask_0.cuda().float()
                loss_0 = 0
                one_hot_mask_1 = torch.nn.functional.one_hot(target_1.long(), num_classes=4).transpose(1, -1).squeeze(
                    -1)
                target_1 = one_hot_mask_1.cuda().float()
                loss_1 = 0
                for k in range(4):
                    loss_0 += criteron(op_0[:, k, :, :], target_0[:, k, :, :], 4)
                    loss_1 += criteron(op_1[:, k, :, :], target_1[:, k, :, :], 4)
                loss_0 = loss_0 / 4
                loss_1 = loss_1 / 4
            else:
                loss_0 = criteron(op_0, target_0)
                loss_1 = criteron(op_1, target_1)


            loss = loss_0 + loss_1
            regressor_loss += float(loss)/2

            output_dm = domain_predictor(features.detach())
            loss_dm = domain_criterion(output_dm, domain_target)

            loss = loss + args["alpha"] * loss_dm
            loss.backward()
            optimizer.step()

            domain_loss += float(loss_dm)

            output_dm_conf = np.argmax(output_dm.detach().cpu().numpy(), axis=1)
            domain_target = np.argmax(domain_target.detach().cpu().numpy(), axis=1)
            true_domains.append(np.array(domain_target))
            pred_domains.append(np.array(output_dm_conf))


            if batch_idx % args["log_interval"] == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\t Regressor Loss: {:.6f}'.format(
                    epoch, (batch_idx+1) * len(data), len(source_train_dataloader.dataset),
                           100. * (batch_idx+1) / len(source_train_dataloader), regressor_loss), flush=True)

            del target
            del features
            del loss

    av_loss = regressor_loss / batches

    av_dom = loss_dm / batches

    true_domains = np.array(true_domains).reshape(-1)
    pred_domains = np.array(pred_domains).reshape(-1)

    acc = accuracy_score(true_domains, pred_domains)

    print('\nTraining set: Average loss: {:.4f}'.format(av_loss,  flush=True))
    print('Training set: Average Dom loss: {:.4f}'.format(av_dom,  flush=True))
    print('Training set: Average Acc: {:.4f}\n'.format(acc,  flush=True))

    return av_loss, acc, av_dom, np.NaN


def val_encoder_domain_unlearn_semi(args, models, val_loaders, criterions, problem):
    cuda = torch.cuda.is_available()

    [encoder, regressor, domain_predictor] = models
    [source_val_dataloader, target_val_dataloader, _, _] = val_loaders
    [criteron, _, _] = criterions

    encoder.eval()
    regressor.eval()
    domain_predictor.eval()

    val_loss = 0

    true_domains = []
    pred_domains = []

    batches = 0
    with torch.no_grad():
        for batch_idx, (source_load, target_load) in enumerate(zip(source_val_dataloader, target_val_dataloader)):
            s_data, s_target, s_domain = source_load['image'], source_load['mask'], source_load['domain']
            t_data, t_target, t_domain = target_load['image'], target_load['mask'], target_load['domain']
            n1 = np.random.randint(1, len(s_data) - 1)
            n2 = len(s_data) - n1

            s_data = s_data[:n1]
            s_target = s_target[:n1]
            s_domain = s_domain[:n1]

            t_data = t_data[:n2]
            t_target = t_target[:n2]
            t_domain = t_domain[:n2]

            data = torch.cat((s_data, t_data), 0)
            target = torch.cat((s_target, t_target), 0)
            domain_target = torch.cat((s_domain, t_domain), 0)

            target = target.type(torch.LongTensor)

            if cuda:
                data, target, domain_target = data.cuda(), target.cuda(), domain_target.cuda()

            data, target, domain_target = Variable(data), Variable(target), Variable(domain_target)

            if list(data.size())[0] == args["batch_size"]:
                batches += 1
                features = encoder(data.unsqueeze(1))
                output_pred = regressor(features)

                op_0 = output_pred[:n1]
                target_0 = target[:n1]

                op_1 = output_pred[n1:]
                target_1 = target[n1:]

                if problem == 'wgc':
                    one_hot_mask_0 = torch.nn.functional.one_hot(target_0.long(), num_classes=4).transpose(1,
                                                                                                           -1).squeeze(
                        -1)
                    target_0 = one_hot_mask_0.cuda().float()
                    loss_0 = 0
                    one_hot_mask_1 = torch.nn.functional.one_hot(target_1.long(), num_classes=4).transpose(1,
                                                                                                           -1).squeeze(
                        -1)
                    target_1 = one_hot_mask_1.cuda().float()
                    loss_1 = 0
                    for k in range(4):
                        loss_0 += criteron(op_0[:, k, :, :], target_0[:, k, :, :], 4)
                        loss_1 += criteron(op_1[:, k, :, :], target_1[:, k, :, :], 4)
                    loss_0 = loss_0 / 4
                    loss_1 = loss_1 / 4
                else:
                    loss_0 = criteron(op_0, target_0)
                    loss_1 = criteron(op_1, target_1)


                loss = loss_0 + loss_1
                val_loss += float(loss)/2

                domains = domain_predictor.forward(features)
                domains = np.argmax(domains.detach().cpu().numpy(), axis=1)
                domain_target = np.argmax(domain_target.detach().cpu().numpy(), axis=1)
                true_domains.append(domain_target)
                pred_domains.append(domains)

    val_loss = val_loss / batches

    true_domains = np.array(true_domains).reshape(-1)
    pred_domains = np.array(pred_domains).reshape(-1)

    val_acc = accuracy_score(true_domains, pred_domains)

    print('\nValidation set: Average loss: {:.4f}\n'.format(val_loss,  flush=True))
    print('Validation set: Average Acc: {:.4f}\n'.format(val_acc,  flush=True))

    return val_loss, val_acc

def train_unlearn_semi(args, models, train_loaders, optimizers, criterions, epoch, problem):
    cuda = torch.cuda.is_available()

    [encoder, regressor, domain_predictor] = models
    [optimizer, optimizer_conf, optimizer_dm] = optimizers
    [source_train_dataloader, target_train_dataloader, source_int_train_dataloader,
     target_int_train_dataloader] = train_loaders
    [criteron, conf_criterion, domain_criterion] = criterions

    regressor_loss = 0
    domain_loss = 0
    conf_loss = 0

    encoder.train()
    regressor.train()
    domain_predictor.train()

    true_domains = []
    pred_domains = []

    batches = 0
    for batch_idx, (source_load, target_load, source_int_load, target_int_load) in enumerate(
            zip(source_train_dataloader, target_train_dataloader, source_int_train_dataloader,
                target_int_train_dataloader)):
        s_data, s_target, s_domain = source_load['image'], source_load['mask'], source_load['domain']
        t_data, t_target, t_domain = target_load['image'], target_load['mask'], target_load['domain']
        s_int_data, s_int_target, s_int_domain = source_int_load['image'], source_int_load['mask'], source_int_load[
            'domain']
        t_int_data, t_int_target, t_int_domain = target_int_load['image'], target_int_load['mask'], target_int_load[
            'domain']
        n1 = np.random.randint(1, len(s_data) - 1)
        n2 = len(s_data) - n1

        s_data = s_data[:n1]
        s_target = s_target[:n1]
        s_domain = s_domain[:n1]

        t_data = t_data[:n2]
        t_target = t_target[:n2]
        t_domain = t_domain[:n2]

        s_int_data = s_int_data[:n1]
        s_int_domain = s_int_domain[:n1]
        t_int_data = t_int_data[:n2]
        t_int_domain = t_int_domain[:n2]

        data = torch.cat((s_data, t_data), 0)
        target = torch.cat((s_target, t_target), 0)
        domain_target = torch.cat((s_domain, t_domain), 0)

        int_data = torch.cat((s_int_data, t_int_data), 0)
        int_domain = torch.cat((s_int_domain, t_int_domain), 0)
        target = target.type(torch.LongTensor)

        if cuda:
            data, target, domain_target, int_data, int_domain = data.cuda(), target.cuda(), domain_target.cuda(), int_data.cuda(), int_domain.cuda()

        data, target, domain_target, int_data, int_domain = Variable(data), Variable(target), Variable(
            domain_target), Variable(int_data), Variable(int_domain)

        if list(data.size())[0] == args["batch_size"]:
            if list(int_domain.size())[0] == args["batch_size"]:

                batches += 1

                # First update the encoder and regressor
                optimizer.zero_grad()
                features = encoder(data.unsqueeze(1))
                output_pred = regressor(features)

                op_0 = output_pred[:n1]
                target_0 = target[:n1]

                op_1 = output_pred[n1:]
                target_1 = target[n1:]

                if problem == 'wgc':
                    one_hot_mask_0 = torch.nn.functional.one_hot(target_0.long(), num_classes=4).transpose(1,
                                                                                                           -1).squeeze(
                        -1)
                    target_0 = one_hot_mask_0.cuda().float()
                    loss_0 = 0
                    one_hot_mask_1 = torch.nn.functional.one_hot(target_1.long(), num_classes=4).transpose(1,
                                                                                                           -1).squeeze(
                        -1)
                    target_1 = one_hot_mask_1.cuda().float()
                    loss_1 = 0
                    for k in range(4):
                        loss_0 += criteron(op_0[:, k, :, :], target_0[:, k, :, :], 4)
                        loss_1 += criteron(op_1[:, k, :, :], target_1[:, k, :, :], 4)
                    loss_0 = loss_0 / 4
                    loss_1 = loss_1 / 4
                else:
                    loss_0 = criteron(op_0, target_0)
                    loss_1 = criteron(op_1, target_1)

                loss_total = (loss_0 + loss_1)/2
                loss_total = Variable(loss_total, requires_grad=True)
                loss_total.backward()
                optimizer.step()

                # Now update just the domain classifier on the intersection data only
                optimizer_dm.zero_grad()
                new_features = encoder(int_data.unsqueeze(1))
                output_dm = domain_predictor(new_features.detach())
                loss_dm = domain_criterion(output_dm, int_domain)
                loss_dm.backward()
                optimizer_dm.step()

                # Now update just the encoder using the domain loss
                optimizer_conf.zero_grad()
                output_dm_conf = domain_predictor(new_features)
                loss_conf = args["beta"] * conf_criterion(output_dm_conf, int_domain)
                loss_conf.backward(retain_graph=False)
                optimizer_conf.step()

                regressor_loss += float(loss_total)
                domain_loss += float(loss_dm)
                conf_loss += float(loss_conf)

                output_dm_conf = np.argmax(output_dm_conf.detach().cpu().numpy(), axis=1)
                domain_target = np.argmax(int_domain.detach().cpu().numpy(), axis=1)
                true_domains.append(np.array(domain_target))
                pred_domains.append(np.array(output_dm_conf))

                if batch_idx % args["log_interval"] == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, (batch_idx+1) * len(data), len(source_train_dataloader.dataset),
                               100. * (batch_idx+1) / len(source_train_dataloader), loss_total.item()), flush=True)
                    print('\t \t Confusion loss = ', loss_conf.item())
                    print('\t \t Domain Loss = ', loss_dm.item(), flush=True)
                del target
                del loss_total
                del features

    av_loss = regressor_loss / batches

    av_conf = loss_conf / batches

    av_dom = loss_dm / batches

    true_domains = np.array(true_domains).reshape(-1)
    pred_domains = np.array(pred_domains).reshape(-1)

    acc = accuracy_score(true_domains, pred_domains)

    print('Training set: Average loss: {:.4f}'.format(av_loss,  flush=True))
    print('Training set: Average Conf loss: {:.4f}'.format(av_conf,  flush=True))
    print('Training set: Average Dom loss: {:.4f}'.format(av_dom,  flush=True))

    print('Training set: Average Acc: {:.4f}\n'.format(acc,  flush=True))

    return av_loss, acc, av_dom, av_conf


def val_unlearn_semi(args, models, val_loaders, criterions, problem):
    cuda = torch.cuda.is_available()

    [encoder, regressor, domain_predictor] = models
    [source_val_dataloader, target_val_dataloader, source_int_val_dataloader, target_int_val_dataloader] = val_loaders

    [criteron, _, _] = criterions

    encoder.eval()
    regressor.eval()
    domain_predictor.eval()

    val_loss = 0

    true_domains = []
    pred_domains = []

    batches = 0
    with torch.no_grad():
        for batch_idx, (source_load, target_load, source_int_load, target_int_load) in enumerate(
                zip(source_val_dataloader, target_val_dataloader, source_int_val_dataloader,
                    target_int_val_dataloader)):
            s_data, s_target, s_domain = source_load['image'], source_load['mask'], source_load['domain']
            t_data, t_target, t_domain = target_load['image'], target_load['mask'], target_load['domain']
            s_int_data, s_int_target, s_int_domain = source_int_load['image'], source_int_load['mask'], source_int_load[
                'domain']
            t_int_data, t_int_target, t_int_domain = target_int_load['image'], target_int_load['mask'], target_int_load[
                'domain']
            n1 = np.random.randint(1, len(s_data) - 1)
            n2 = len(s_data) - n1

            s_data = s_data[:n1]
            s_target = s_target[:n1]
            s_domain = s_domain[:n1]

            t_data = t_data[:n2]
            t_target = t_target[:n2]
            t_domain = t_domain[:n2]

            s_int_data = s_int_data[:n1]
            s_int_domain = s_int_domain[:n1]
            t_int_data = t_int_data[:n2]
            t_int_domain = t_int_domain[:n2]

            data = torch.cat((s_data, t_data), 0)
            target = torch.cat((s_target, t_target), 0)
            domain_target = torch.cat((s_domain, t_domain), 0)

            int_data = torch.cat((s_int_data, t_int_data), 0)
            int_domain = torch.cat((s_int_domain, t_int_domain), 0)
            target = target.type(torch.LongTensor)

            if cuda:
                data, target, domain_target, int_data, int_domain = data.cuda(), target.cuda(), domain_target.cuda(), int_data.cuda(), int_domain.cuda()

            data, target, domain_target, int_data, int_domain = Variable(data), Variable(target), Variable(
                domain_target), Variable(int_data), Variable(int_domain)

            if list(data.size())[0] == args["batch_size"]:
                if list(int_data.size())[0] == args["batch_size"]:
                    batches += 1
                    features = encoder(data.unsqueeze(1))
                    output_pred = regressor(features)

                    op_0 = output_pred[:n1]
                    target_0 = target[:n1]

                    op_1 = output_pred[n1:]
                    target_1 = target[n1:]

                    if problem == 'wgc':
                        one_hot_mask_0 = torch.nn.functional.one_hot(target_0.long(), num_classes=4).transpose(1,
                                                                                                               -1).squeeze(
                            -1)
                        target_0 = one_hot_mask_0.cuda().float()
                        loss_0 = 0
                        one_hot_mask_1 = torch.nn.functional.one_hot(target_1.long(), num_classes=4).transpose(1,
                                                                                                               -1).squeeze(
                            -1)
                        target_1 = one_hot_mask_1.cuda().float()
                        loss_1 = 0
                        for k in range(4):
                            loss_0 += criteron(op_0[:, k, :, :], target_0[:, k, :, :], 4)
                            loss_1 += criteron(op_1[:, k, :, :], target_1[:, k, :, :], 4)
                        loss_0 = loss_0 / 4
                        loss_1 = loss_1 / 4
                    else:
                        loss_0 = criteron(op_0, target_0)
                        loss_1 = criteron(op_1, target_1)

                    loss_total = loss_0 + loss_1
                    val_loss += float(loss_total)/2

                    new_features = encoder(int_data.unsqueeze(1))
                    domains = domain_predictor.forward(new_features)
                    domains = np.argmax(domains.detach().cpu().numpy(), axis=1)
                    domain_target = np.argmax(int_domain.detach().cpu().numpy(), axis=1)
                    true_domains.append(domain_target)
                    pred_domains.append(domains)

    val_loss = val_loss / batches

    true_domains = np.array(true_domains).reshape(-1)
    pred_domains = np.array(pred_domains).reshape(-1)

    acc = accuracy_score(true_domains, pred_domains)

    print('\nValidation set: Average loss: {:.4f}\n'.format(val_loss,  flush=True))
    print('Validation set: Average Acc: {:.4f}\n'.format(acc,  flush=True))

    return val_loss, acc


def cmd_train(ctx):
    cuda = torch.cuda.is_available()
    source_domain = torch.Tensor([0, 1])
    target_domain = torch.Tensor([1, 0])
    problem = ctx["problem"]
    out_channels = 4 if problem == 'wgc' else 1
    img_pth = 'images_wgc' if problem == 'wgc' else 'images'
    msk_pth = 'masks_wgc' if problem == 'wgc' else 'masks'
    batch_size = ctx["batch_size"]
    patience = ctx["patience"]
    out_dir = ctx["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    for f in os.listdir(out_dir):
        os.remove(out_dir + f)
    num_samples = 10
    source_train_dataloader = get_dataloader(os.path.join(ctx["source_train_dir"], img_pth),
                                             os.path.join(ctx["source_train_dir"], msk_pth), batch_size,
                                             None, domain=source_domain)
    source_val_dataloader = get_dataloader(os.path.join(ctx["source_val_dir"], img_pth),
                                           os.path.join(ctx["source_val_dir"], msk_pth),
                                           batch_size, None, domain=source_domain)

    target_train_dataloader = get_dataloader(os.path.join(ctx["target_train_dir"], img_pth),
                                             os.path.join(ctx["target_train_dir"], msk_pth), batch_size, None,
                                             shuffle=True, domain=target_domain)

    target_val_dataloader = get_dataloader(os.path.join(ctx["target_val_dir"], img_pth),
                                           os.path.join(ctx["target_val_dir"], msk_pth), batch_size, None,
                                           shuffle=False, drop_last=False, domain=target_domain)

    target_train_dataloader_int = get_dataloader(os.path.join(ctx["target_train_dir"], img_pth),
                                             os.path.join(ctx["target_train_dir"], msk_pth), batch_size, None,
                                             shuffle=True, domain=target_domain)

    target_val_dataloader_int = get_dataloader(os.path.join(ctx["target_val_dir"], img_pth),
                                           os.path.join(ctx["target_val_dir"], msk_pth), batch_size, None,
                                           shuffle=False, drop_last=False, domain=target_domain)



    # Load the model
    unet = UNet()
    segmenter = Segmenter(out_channels=out_channels)
    domain_pred = DomainPredictor(2)

    if cuda:
        unet = unet.cuda()
        segmenter = segmenter.cuda()
        domain_pred = domain_pred.cuda()

    # Make everything parallelisable
    unet = nn.DataParallel(unet)
    segmenter = nn.DataParallel(segmenter)
    domain_pred = nn.DataParallel(domain_pred)

    criteron = dice_loss()
    criteron.cuda()
    domain_criterion = nn.BCELoss()
    domain_criterion.cuda()
    conf_criterion = confusion_loss()
    conf_criterion.cuda()

    optimizer_step1 = optim.Adam(
        list(unet.parameters()) + list(segmenter.parameters()) + list(domain_pred.parameters()), lr=ctx["stage1_lr"])
    optimizer = optim.Adam(list(unet.parameters()) + list(segmenter.parameters()), lr=1e-4)
    optimizer_conf = optim.Adam(list(unet.parameters()), lr=1e-4)
    optimizer_dm = optim.Adam(list(domain_pred.parameters()), lr=1e-4)  # Lower learning rate for the unlearning bit

    # Initalise the early stopping
    early_stopping = EarlyStoppingUnlearning(patience, verbose=False)

    if ctx["resume"]:
        checkpoint = torch.load(ctx["resume_path"])        
        epoch_reached = checkpoint["epoch"]
        unet.load_state_dict(checkpoint['unet'])
        segmenter.load_state_dict(checkpoint['segmenter'])
        domain_pred.load_state_dict(checkpoint['domain_pred'])
        optimizer_step1.load_state_dict(checkpoint['optimizer_step1'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        optimizer_conf.load_state_dict(checkpoint['optimizer_conf'])
        optimizer_dm.load_state_dict(checkpoint['optimizer_dm'])

        print("Loaded checkpoint at epoch {}".format(epoch_reached))
        wandb_id = checkpoint["wandb_id"]
        wandb.init(project="Domain Unlearn Supervised", entity="autoda", resume="allow", id=wandb_id,
                    config={"alpha": ctx["alpha"], "beta": ctx["beta"], "source": ctx["source"], "target": ctx["target"], "problem": ctx["problem"]}
                    )

    else:
        wandb_id = wandb.util.generate_id()
        wandb.init(project="Domain Unlearn Semi-Supervised", entity="autoda", resume="allow", id=wandb_id,
                    config={"alpha": ctx["alpha"], "beta": ctx["beta"], "source": ctx["source"], "target": ctx["target"], "problem": ctx["problem"], "stage1_lr": ctx["stage1_lr"]}
                    )
        epoch_reached = 0

    models = [unet, segmenter, domain_pred]
    optimizers_stage2 = [optimizer, optimizer_conf, optimizer_dm]
    train_dataloaders = [source_train_dataloader, target_train_dataloader, source_train_dataloader, target_train_dataloader_int]
    val_dataloaders = [source_val_dataloader, target_val_dataloader, source_val_dataloader, target_val_dataloader_int]
    criterions = [criteron, conf_criterion, domain_criterion]

    epochs = ctx["epochs"]
    epoch_stage_1 = ctx["epoch_stage_1"]

    for epoch in range(epoch_reached, epochs + 1):
        if epoch < epoch_stage_1:
            print('Training Main Encoder')
            print('Epoch ', epoch, '/', epochs, flush=True)
            optimizers_stage1 = [optimizer_step1]
            loss, acc, dm_loss, conf_loss = train_encoder_domain_unlearn_semi(ctx, models, train_dataloaders,
                                                                              optimizers_stage1, criterions, epoch, problem)
            dm_loss = dm_loss.detach().cpu().clone().numpy()

            torch.cuda.empty_cache()  # Clear memory cache
            val_loss, val_acc = val_encoder_domain_unlearn_semi(ctx, models, val_dataloaders, criterions, problem)

            wandb.log({'epoch': epoch, 'Train stage 1: Dice loss': loss,
            'Train stage 1: Disc loss': dm_loss, 'Train stage 1: Accuracy': acc, 
            'Val stage 1: Dice loss': val_loss, 'Val stage 1: Accuracy': val_acc})            

            if epoch == epoch_stage_1 - 1 or epoch % ctx["checkpoint"]:
                torch.save({
                            'epoch': epoch,
                            'unet': unet.state_dict(),
                            'segmenter': segmenter.state_dict(),
                            'domain_pred': domain_pred.state_dict(),
                            'optimizer_step1': optimizer_step1.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'optimizer_conf': optimizer_conf.state_dict(),
                            'optimizer_dm': optimizer_dm.state_dict(),
                            'wandb_id': wandb_id
                            }, out_dir + '/checkpoint_epoch' + str(epoch) + '.pth')
                wandb.save(out_dir + '/checkpoint_epoch' + str(epoch) + '.pth')

        else:
            print('Unlearning')
            print('Epoch ', epoch, '/', epochs, flush=True)
            torch.cuda.empty_cache()  # Clear memory cache
            loss, acc, dm_loss, conf_loss = train_unlearn_semi(ctx, models, train_dataloaders, optimizers_stage2, criterions, epoch, problem)
            val_loss, val_acc = val_unlearn_semi(ctx, models, val_dataloaders, criterions, problem)

            wandb.log({'epoch': epoch, 'Train stage 2: Dice loss': loss, 'Train stage 1: Conf loss': conf_loss,
            'Train stage 2: Disc loss': dm_loss, 'Train stage 2: Accuracy': acc, 
            'Val stage 2: Dice loss': val_loss, 'Val stage 2: Accuracy': val_acc})            

            # Decide whether the model should stop training or not
            early_stopping(val_loss, models, epoch, optimizers_stage2, loss,
                           [CHK_PATH_UNET, CHK_PATH_SEGMENTER, CHK_PATH_DOMAIN], out_dir, wandb_id)
            if early_stopping.early_stop:
                sys.exit('Patience Reached - Early Stopping Activated')

            if epoch == epochs:
                print('Finished Training', flush=True)
                print('Saving the model', flush=True)

            torch.save({
                        'epoch': epoch,
                        'unet': unet.state_dict(),
                        'segmenter': segmenter.state_dict(),
                        'domain_pred': domain_pred.state_dict(),
                        'optimizer_step1': optimizer_step1.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'optimizer_conf': optimizer_conf.state_dict(),
                        'optimizer_dm': optimizer_dm.state_dict(),
                        'wandb_id': wandb_id
                        }, out_dir + '/checkpoint_epoch' + str(epoch) + '.pth')
            wandb.save(out_dir + '/checkpoint_epoch' + str(epoch) + '.pth')

            torch.cuda.empty_cache()  # Clear memory cache