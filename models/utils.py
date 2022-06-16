import numpy as np
import torch
import torch.nn as nn

from metrics.dice import dice_score


class EarlyStopping:
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """

    def __init__(self, patience=5, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True


def cosine_rampdown(current, rampdown_length):
    """Cosine rampdown from https://arxiv.org/abs/1608.03983"""
    assert 0 <= current <= rampdown_length
    return float(.5 * (np.cos(np.pi * current / rampdown_length) + 1))


def cosine_lr(current_epoch, num_epochs, initial_lr):
    return initial_lr * cosine_rampdown(current_epoch, num_epochs)


def sigmoid_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def scheduler(epoch, lr):
    if epoch % 5 == 0 and epoch != 0:
        lr = lr / 2
    return lr


def get_current_consistency_weight(weight, epoch, rampup):
    """Consistency ramp-up from https://arxiv.org/abs/1610.02242"""
    return weight * sigmoid_rampup(epoch, rampup)


def validation(model, loader, writer, epoch, out_channels, num_labels):
    val_loss = 0.0

    num_samples = 0
    num_steps = 0

    for i, batch in enumerate(loader):
        image_data, mask_data = batch['image'], batch['mask']
        if out_channels != 1:
            one_hot_mask = torch.nn.functional.one_hot(mask_data.long(), num_classes=out_channels).squeeze(-1)
            mask_data_gpu = one_hot_mask.cuda().float()
        else:
            mask_data_gpu = mask_data.cuda()
        image_data_gpu = image_data.cuda()

        loss = 0
        with torch.no_grad():
            model_out = model(image_data_gpu)
            if out_channels != 1:
                for k in range(out_channels):
                    dice_loss = -dice_score(model_out[:, k, :, :], mask_data_gpu[:, :, :, k], num_labels)
                    loss += dice_loss
                dice_loss = loss / out_channels
            else:
                dice_loss = -dice_score(model_out, mask_data_gpu, num_labels)
            val_loss += dice_loss.item()

        predictions = model_out.cpu().numpy()

        num_samples += len(predictions)
        num_steps += 1

    val_loss_avg = val_loss / num_steps

    writer.add_scalars('losses', {'loss': val_loss_avg}, epoch)
    return val_loss_avg


class EarlyStoppingUnlearning:
    # Early stops the training if the validation loss doesnt improve after a given patience
    def __init__(self, patience=5, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model, epoch, optimizer, loss, PTH):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch, optimizer, loss, PTH)
        elif score < self.best_score:
            self.counter += 1
            print('Early Stopping Counter: ', self.counter, '/', self.patience)
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch, optimizer, loss, PTH)
            self.counter = 0

    def save_checkpoint(self, val_loss, models, epoch, optimizer, loss, PTHS):
        # Saves the model when the validation loss decreases
        if self.verbose:
            print('Validation loss decreased: ', self.val_loss_min, ' --> ', val_loss, 'Saving model ...')
        [encoder, regressor, domain_predictor] = models
        [PATH_ENCODER, PATH_REGRESSOR, PATH_DOMAIN] = PTHS
        if PATH_ENCODER:
            torch.save(encoder.state_dict(), PATH_ENCODER)
        if PATH_REGRESSOR:
            torch.save(regressor.state_dict(), PATH_REGRESSOR)
        if PATH_DOMAIN:
            torch.save(domain_predictor.state_dict(), PATH_DOMAIN)


class Args:
    # Store lots of the parameters that we might need to train the model
    def __init__(self):
        self.batch_size = 8
        self.log_interval = 10
        self.learning_rate = 1e-4
        self.epochs = 2
        self.train_val_prop = 0.9
        self.patience = 5
        self.channels_first = True
        self.diff_model_flag = False
        self.alpha = 1
        self.beta = 10
        self.epoch_stage_1 = 100
        self.epoch_reached = 1


class confusion_loss(nn.Module):
    def __init__(self, task=0):
        super(confusion_loss, self).__init__()
        self.task = task

    def forward(self, x, target):
        # We only care about x
        log = torch.log(x)
        log_sum = torch.sum(log, dim=1)
        normalised_log_sum = torch.div(log_sum, x.size()[1])
        loss = torch.mul(torch.sum(normalised_log_sum, dim=0), -1)
        return loss
