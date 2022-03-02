# Nicola Dinsdale 2020
# Unlearning main for the segmentation model
########################################################################################################################
########################################################################################################################
# Create an args class
import os.path
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from torch.autograd import Variable

from config.io import *
from data.utils import get_dataloader
from metrics.confusion_loss import confusion_loss
from metrics.dice import dice_loss
from models.unlearn import UNet, Segmenter, DomainPredictor
from models.utils import EarlyStoppingUnlearning


def train_encoder_unlearn(args, models, train_loaders, optimizers, criterions, epoch):
    cuda = torch.cuda.is_available()

    [encoder, regressor, domain_predictor] = models
    [optimizer] = optimizers
    [source_train_dataloader, target_train_dataloader] = train_loaders
    [criteron, _, domain_criterion] = criterions
    regressor_loss = 0
    domain_loss = 0

    encoder.train()
    regressor.train()
    domain_predictor.train()

    true_domains = []
    pred_domains = []

    batch_size = args["batch_size"]
    batches = 0
    for batch_idx, (source_load, target_load) in enumerate(zip(source_train_dataloader, target_train_dataloader)):
        s_data, s_target, s_domain = source_load['image'], source_load['mask'], source_load['domain']
        t_data, t_target, t_domain = target_load['image'], target_load['mask'], target_load['domain']
        if len(s_data) == batch_size:
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

            if list(data.size())[0] == batch_size:
                batches += 1

                # First update the encoder and regressor
                optimizer.zero_grad()
                data = torch.unsqueeze(data, 1)
                features = encoder(data)
                output_pred = regressor(features)

                op_0 = output_pred[:n1]
                target_0 = target[:n1]
                loss_0 = criteron(op_0, target_0)

                op_1 = output_pred[n1:]
                target_1 = target[n1:]
                loss_1 = criteron(op_1, target_1)

                r_loss = loss_0 + loss_1
                domain_pred = domain_predictor(features)

                d_loss = domain_criterion(domain_pred, domain_target)
                loss = r_loss + d_loss
                loss.backward()
                optimizer.step()

                regressor_loss += r_loss
                domain_loss += d_loss

                domains = np.argmax(domain_pred.detach().cpu().numpy(), axis=1)
                domain_target = np.argmax(domain_target.detach().cpu().numpy(), axis=1)
                true_domains.append(domain_target)
                pred_domains.append(domains)

                if batch_idx % args["log_interval"] == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\t Regressor Loss: {:.6f}'.format(
                        epoch, (batch_idx + 1) * len(data), len(source_train_dataloader.dataset),
                               100. * (batch_idx + 1) / len(source_train_dataloader), r_loss.item()), flush=True)
                    print('Regressor Loss: {:.4f}'.format(r_loss, flush=True))
                    print('Domain Loss: {:.4f}'.format(d_loss, flush=True))

                del target
                del r_loss
                del d_loss
                del features

    av_loss = regressor_loss / batches

    av_dom_loss = domain_loss / batches

    true_domains = np.array(true_domains).reshape(-1)
    pred_domains = np.array(pred_domains).reshape(-1)
    acc = accuracy_score(true_domains, pred_domains)

    print('\nTraining set: Average loss: {:.4f}'.format(av_loss, flush=True))
    print('Training set: Average Domain loss: {:.4f}'.format(av_dom_loss, flush=True))
    print('Training set: Average Acc: {:.4f}'.format(acc, flush=True))

    return av_loss, acc, av_dom_loss, np.NaN


def val_encoder_unlearn(args, models, val_loaders, criterions):
    cuda = torch.cuda.is_available()

    [encoder, regressor, domain_predictor] = models
    [source_val_dataloader, target_val_dataloader] = val_loaders
    [criteron, _, domain_criterion] = criterions

    encoder.eval()
    regressor.eval()
    domain_predictor.eval()

    regressor_loss = 0
    domain_loss = 0

    true_domains = []
    pred_domains = []

    batches = 0
    with torch.no_grad():
        for batch_idx, (source_load, target_load) in enumerate(zip(source_val_dataloader, target_val_dataloader)):
            s_data, s_target, s_domain = source_load['image'], source_load['mask'], source_load['domain']
            t_data, t_target, t_domain = target_load['image'], target_load['mask'], target_load['domain']

            if len(s_data) == args["batch_size"]:

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
                    data = torch.unsqueeze(data, 1)
                    features = encoder(data)
                    output_pred = regressor(features)

                    op_0 = output_pred[:n1]
                    target_0 = target[:n1]
                    loss_0 = criteron(op_0, target_0)

                    op_1 = output_pred[n1:]
                    target_1 = target[n1:]
                    loss_1 = criteron(op_1, target_1)

                    r_loss = loss_0 + loss_1

                    domain_pred = domain_predictor(features)

                    d_loss = domain_criterion(domain_pred, domain_target)

                    domains = np.argmax(domain_pred.detach().cpu().numpy(), axis=1)
                    domain_target = np.argmax(domain_target.detach().cpu().numpy(), axis=1)
                    true_domains.append(domain_target)
                    pred_domains.append(domains)

                    regressor_loss += r_loss
                    domain_loss += d_loss

    val_loss = regressor_loss / batches

    dom_loss = domain_loss / batches

    true_domains = np.array(true_domains).reshape(-1)
    pred_domains = np.array(pred_domains).reshape(-1)
    acc = accuracy_score(true_domains, pred_domains)

    print('\nValidation set: Average loss: {:.4f}\n'.format(val_loss, flush=True))
    print('Validation set: Average Domain loss: {:.4f}\n'.format(dom_loss, flush=True))
    print(' Validation set: Average Acc: {:.4f}'.format(acc, flush=True))

    return val_loss, acc


def train_unlearn(args, models, train_loaders, optimizers, criterions, epoch):
    cuda = torch.cuda.is_available()

    [encoder, regressor, domain_predictor] = models
    [optimizer, optimizer_conf, optimizer_dm] = optimizers
    [source_train_dataloader, target_train_dataloader] = train_loaders
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

    for batch_idx, (source_load, target_load) in enumerate(zip(source_train_dataloader, target_train_dataloader)):
        s_data, s_target, s_domain = source_load['image'], source_load['mask'], source_load['domain']
        t_data, t_target, t_domain = target_load['image'], target_load['mask'], target_load['domain']

        if len(s_data) == args["batch_size"]:

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

                # First update the encoder and regressor
                optimizer.zero_grad()
                data = torch.unsqueeze(data, 1)
                features = encoder(data)
                output_pred = regressor(features)

                op_0 = output_pred[:n1]
                target_0 = target[:n1]
                loss_0 = criteron(op_0, target_0)

                op_1 = output_pred[n1:]
                target_1 = target[n1:]
                loss_1 = criteron(op_1, target_1)

                loss_total = loss_0 + loss_1
                loss_total.backward(retain_graph=True)
                # optimizer.step()
                optimizer.zero_grad()
                # Now update just the domain classifier
                optimizer_dm.zero_grad()
                output_dm = domain_predictor(features.detach())

                loss_dm = domain_criterion(output_dm, domain_target)
                loss_dm.backward(retain_graph=False)
                optimizer_dm.step()

                # Now update just the encoder using the domain loss
                optimizer_conf.zero_grad()
                output_dm_conf = domain_predictor(features)
                loss_conf = args["beta"] * conf_criterion(output_dm_conf, domain_target)
                loss_conf.backward(retain_graph=False)
                optimizer_conf.step()

                regressor_loss += loss_total
                domain_loss += loss_dm
                conf_loss += loss_conf

                output_dm_conf = np.argmax(output_dm_conf.detach().cpu().numpy(), axis=1)
                domain_target = np.argmax(domain_target.detach().cpu().numpy(), axis=1)
                true_domains.append(domain_target)
                pred_domains.append(output_dm_conf)

                if batch_idx % args["log_interval"] == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, (batch_idx + 1) * len(data), len(source_train_dataloader.dataset),
                               100. * (batch_idx + 1) / len(source_train_dataloader), loss_total.item()), flush=True)
                    print('\t \t Confusion loss = ', loss_conf.item())
                    print('\t \t Domain Loss = ', loss_dm.item(), flush=True)
                del target
                del loss_total
                del features

            torch.cuda.empty_cache()  # Clear memory cache

    av_loss = regressor_loss / batches

    av_conf = conf_loss / batches

    av_dom = domain_loss / batches

    true_domains = np.array(true_domains).reshape(-1)
    pred_domains = np.array(pred_domains).reshape(-1)
    acc = accuracy_score(true_domains, pred_domains)

    print('\nTraining set: Average loss: {:.4f}'.format(av_loss, flush=True))
    print('\nTraining set: Average Conf loss: {:.4f}'.format(av_conf, flush=True))
    print('\nTraining set: Average Dom loss: {:.4f}'.format(av_dom, flush=True))

    print('\nTraining set: Average Acc: {:.4f}\n'.format(acc, flush=True))

    return av_loss, acc, av_dom, av_conf


def val_unlearn(args, models, val_loaders, criterions):
    cuda = torch.cuda.is_available()

    [encoder, regressor, domain_predictor] = models
    [source_val_dataloader, target_val_dataloader] = val_loaders
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
            if len(s_data) == args["batch_size"]:

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

                    features = encoder(data)
                    output_pred = regressor(features)
                    print('output_pred:', output_pred.shape)

                    op_0 = output_pred[:n1]
                    target_0 = target[:n1]
                    loss_0 = criteron(op_0, target_0)

                    op_1 = output_pred[n1:]
                    target_1 = target[n1:]
                    loss_1 = criteron(op_1, target_1)

                    loss_total = loss_0 + loss_1
                    val_loss += loss_total

                    domains = domain_predictor.forward(features)
                    domains = np.argmax(domains.detach().cpu().numpy(), axis=1)
                    domain_target = np.argmax(domain_target.detach().cpu().numpy(), axis=1)
                    true_domains.append(domain_target)
                    pred_domains.append(domains)

            torch.cuda.empty_cache()

    val_loss = val_loss / batches

    true_domains = np.array(true_domains).reshape(-1)
    pred_domains = np.array(pred_domains).reshape(-1)
    acc = accuracy_score(true_domains, pred_domains)

    print('\nValidation set: Average loss: {:.4f}\n'.format(val_loss, flush=True))
    print('Validation set: Average Acc: {:.4f}\n'.format(acc, flush=True))

    return val_loss, acc


def cmd_train(ctx):
    cuda = torch.cuda.is_available()
    source_domain = torch.Tensor([0, 1])
    target_domain = torch.Tensor([1, 0])
    problem = ctx["problem"]
    img_pth = 'images_wgc' if problem == 'wgc' else 'images'
    msk_pth = 'masks_wgc' if problem == 'wgc' else 'masks'
    batch_size = ctx["batch_size"]
    out_dir = ctx["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
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

    # Load the model
    u_net = UNet()
    segmenter = Segmenter()
    domain_pred = DomainPredictor(2)

    if cuda:
        u_net = u_net.cuda()
        segmenter = segmenter.cuda()
        domain_pred = domain_pred.cuda()

    # Make everything parallelisable
    u_net = nn.DataParallel(u_net)
    segmenter = nn.DataParallel(segmenter)
    domain_pred = nn.DataParallel(domain_pred)

    criterion = dice_loss()
    criterion.cuda()
    domain_criterion = nn.BCELoss()
    domain_criterion.cuda()
    conf_criterion = confusion_loss()
    conf_criterion.cuda()

    optimizer_step1 = optim.Adam(
        list(u_net.parameters()) + list(segmenter.parameters()) + list(domain_pred.parameters()),
        lr=ctx["learning_rate"])
    optimizer = optim.Adam(list(u_net.parameters()) + list(segmenter.parameters()), lr=1e-4)
    optimizer_conf = optim.Adam(list(u_net.parameters()), lr=1e-4)
    optimizer_dm = optim.Adam(list(domain_pred.parameters()), lr=1e-4)  # Lower learning rate for the unlearning bit
    # Initalise the early stopping
    early_stopping = EarlyStoppingUnlearning(ctx["patience"], verbose=False)

    loss_store = []

    models = [u_net, segmenter, domain_pred]
    optimizers = [optimizer, optimizer_conf, optimizer_dm]
    train_dataloaders = [source_train_dataloader, target_train_dataloader]
    val_dataloaders = [source_val_dataloader, target_val_dataloader]
    criterions = [criterion, conf_criterion, domain_criterion]
    epochs = ctx["epochs"]
    epoch_reached = ctx["epoch_reached"]
    epoch_stage_1 = ctx["epoch_stage_1"]
    for epoch in range(epoch_reached, epochs + 1):
        if epoch < epoch_stage_1:
            print('Training Main Encoder')
            print('Epoch ', epoch, '/', epochs, flush=True)
            optimizers = [optimizer_step1]
            loss, acc, dm_loss, conf_loss = train_encoder_unlearn(ctx, models, train_dataloaders, optimizers,
                                                                  criterions, epoch)
            loss = loss.detach().cpu().clone().numpy()
            dm_loss = dm_loss.detach().cpu().clone().numpy()
            torch.cuda.empty_cache()  # Clear memory cache
            val_loss, val_acc = val_encoder_unlearn(ctx, models, val_dataloaders, criterions)
            val_loss = val_loss.detach().cpu().clone().numpy()
            loss_store.append([loss, val_loss, acc, val_acc, dm_loss, conf_loss])

            # Save the losses each epoch so we can plot them live
            np.save(os.path.join(out_dir, LOSS_PATH), np.array(loss_store))

            if epoch == epoch_stage_1 - 1 or epoch % ctx["checkpoint"]:
                torch.save(u_net.state_dict(), os.path.join(out_dir, PRETRAIN_UNET))
                torch.save(segmenter.state_dict(), os.path.join(out_dir, PRETRAIN_SEGMENTER))
                torch.save(domain_pred.state_dict(), os.path.join(out_dir, PRETRAIN_DOMAIN))

        else:
            optimizer = optim.Adam(list(u_net.parameters()) + list(segmenter.parameters()), lr=1e-5)
            optimizer_conf = optim.Adam(list(u_net.parameters()), lr=1e-6)
            optimizer_dm = optim.Adam(list(domain_pred.parameters()), lr=1e-6)
            optimizers = [optimizer, optimizer_conf, optimizer_dm]

            print('Unlearning')
            print('Epoch ', epoch, '/', epochs, flush=True)
            loss, acc, dm_loss, conf_loss = train_unlearn(ctx, models, train_dataloaders, optimizers, criterions,
                                                          epoch)
            torch.cuda.empty_cache()  # Clear memory cache
            val_loss, val_acc = val_unlearn(ctx, models, val_dataloaders, criterions)

            loss_store.append([loss, val_loss, acc, val_acc, dm_loss, conf_loss])
            np.save(os.path.join(out_dir, LOSS_PATH), np.array(loss_store))

            # Decide whether the model should stop training or not
            early_stopping(val_loss, models, epoch, optimizer, loss,
                           [CHK_PATH_UNET, CHK_PATH_SEGMENTER, CHK_PATH_DOMAIN])
            if early_stopping.early_stop:
                loss_store = np.array(loss_store)
                np.save(os.path.join(out_dir, LOSS_PATH), loss_store)
                sys.exit('Patience Reached - Early Stopping Activated')

            if epoch == epochs:
                print('Finished Training', flush=True)
                print('Saving the model', flush=True)

                # Save the model in such a way that we can continue training later
                torch.save(u_net.state_dict(), os.path.join(out_dir, PATH_UNET))
                torch.save(segmenter.state_dict(), os.path.join(out_dir, PATH_SEGMENTER))
                torch.save(domain_pred.state_dict(), os.path.join(out_dir, PATH_DOMAIN))

                loss_store = np.array(loss_store)
                np.save(os.path.join(out_dir, LOSS_PATH), loss_store)

            torch.cuda.empty_cache()  # Clear memory cache
