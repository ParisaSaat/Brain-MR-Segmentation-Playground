from collections import defaultdict

import numpy as np
import torch


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


def dice_score(pred, target):
    eps = 0.0001
    iflat = pred.reshape(-1)
    tflat = target.reshape(-1)
    intersection = (iflat * tflat).sum()
    union = iflat.sum() + tflat.sum()

    dice = (2.0 * intersection + eps) / (union + eps)

    return - dice


def validation(model, loader, writer, metric_fns, epoch, val_samples_dir, out_channels, experiment_name, one_hot=False):
    val_loss = 0.0

    num_samples = 0
    num_steps = 0

    result_dict = defaultdict(list)
    for i, batch in enumerate(loader):
        image_data, mask_data = batch['image'], batch['mask']
        if one_hot:
            one_hot_mask = torch.nn.functional.one_hot(mask_data.long(), num_classes=out_channels).squeeze(-1)
            mask_data_gpu = one_hot_mask.cuda().float()
        else:
            mask_data_gpu = mask_data.cuda()
        image_data_gpu = image_data.cuda()

        loss = 0
        with torch.no_grad():
            model_out = model(image_data_gpu)
            if one_hot:
                for k in range(out_channels):
                    loss += dice_score(model_out[:, k, :, :], mask_data_gpu[:, :, :, k])
                dice_loss = loss / out_channels
            else:
                dice_loss = dice_score(model_out, mask_data_gpu)
            val_loss += dice_loss.item()

        masks = mask_data_gpu.cpu().numpy().astype(np.uint8)
        imgs = image_data_gpu.cpu().numpy().astype(np.uint8)
        predictions = model_out.cpu().numpy()

        for metric_fn in metric_fns:
            for prediction, mask, img in zip(predictions, masks, imgs):
                if one_hot:
                    sum = 0
                    for k in range(out_channels):
                        sum += metric_fn(prediction[k, :, :], mask[:, :, k])
                    res = sum / out_channels
                else:
                    prediction = prediction.squeeze(axis=0)
                    res = metric_fn(prediction, mask)
                dict_key = 'val_{}'.format(metric_fn.__name__)
                if not res or np.isnan(res):
                    continue
                result_dict[dict_key].append(res)
                # chance = random.uniform(0, 1)
                # if chance < PLOTTING_RATE:
                #     plt.imshow(np.rot90(img), cmap='Greys_r')
                #     plt.imshow(np.rot90(prediction) > 0.5, cmap='jet', alpha=0.5)
                #     plt.axis("off")
                #     plt.savefig(val_samples_dir + '/{}_{}_pred.png'.format(epoch, chance))
                #     plt.imshow(np.rot90(imgg), cmap='Greys_r')
                #     plt.imshow(np.rot90(mask) > 0.5, cmap='jet', alpha=0.5)
                #     plt.axis("off")
                #     plt.savefig(val_samples_dir + '/{}_{}_mask.png'.format(epoch, chance))

        num_samples += len(predictions)
        num_steps += 1

    val_loss_avg = val_loss / num_steps

    metrics_dict = defaultdict()
    for key, val in result_dict.items():
        metric_file = '{}_{}'.format(key, experiment_name)
        values = np.asarray(val, dtype=np.float32)
        np.save(metric_file, values)
        mean = np.mean(values)
        std = np.std(values)
        metrics_dict['{}_mean'.format(key)] = mean
        metrics_dict['{}_std'.format(key)] = std
    np.save('metrics_{}.npy'.format(experiment_name), metrics_dict)
    writer.add_scalars('losses', {'loss': val_loss_avg}, epoch)
    writer.add_scalars('metrics', metrics_dict, epoch)
    return val_loss_avg
