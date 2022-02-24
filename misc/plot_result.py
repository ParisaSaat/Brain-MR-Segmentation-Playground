import matplotlib.pyplot as plt
import numpy as np


def plot_segmentation(image, pred_mask, gt_mask, dir, id):
    plt.imshow(np.rot90(image), cmap='Greys_r')
    plt.imshow(np.rot90(pred_mask) > 0.5, cmap='jet', alpha=0.5)
    plt.axis("off")
    plt.savefig(dir + '/{}_pred.png'.format(id))
    plt.imshow(np.rot90(image), cmap='Greys_r')
    plt.imshow(np.rot90(gt_mask) > 0.5, cmap='jet', alpha=0.5)
    plt.axis("off")
    plt.savefig(dir + '/{}_mask.png'.format(id))
