from models.utils_codagan import get_all_data_loaders, prepare_sub_folder, write_html, write_loss, get_config, write_2images, norm
import argparse
from torch.autograd import Variable
from models.codagan import MUNIT_Trainer, UNIT_Trainer
import torch.backends.cudnn as cudnn
import torch
try:
    from itertools import izip as zip
except ImportError: # Will be 3.x series.
    pass
import os
import sys
import math
import shutil
import numpy as np
import wandb
from skimage import io

# parser = argparse.ArgumentParser()
# parser.add_argument('--config', type=str, default="/home/muhammadyusuf.hassan/Brain-MR-Segmentation-Playground/experiments/CXR_lungs_MUNIT_1.0.yaml", help='Path to the config file.')
# parser.add_argument('--output_path', type=str, default="/home/muhammadyusuf.hassan/Brain-MR-Segmentation-Playground/", help="Outputs path.")
# parser.add_argument('--resume', type=int, default=-1)
# parser.add_argument('--snapshot_dir', type=str, default='.')
# opts = parser.parse_args()

cudnn.benchmark = True

def cmd_train(config):
    # Load experiment setting.
    # config = get_config(opts.config)
    # print(config, flush=True)
    display_size = config['display_size']
    config['vgg_model_path'] = config['output_path']

    # Setup model and data loader.
    if config['trainer'] == 'MUNIT':
        trainer = MUNIT_Trainer(config, resume_epoch=config['resume'], snapshot_dir=config['snapshot_dir'])
    elif config['trainer'] == 'UNIT':
        trainer = UNIT_Trainer(config, resume_epoch=config['resume'], snapshot_dir=config['snapshot_dir'])
    else:
        sys.exit("Only support MUNIT|UNIT.")
        os.exit()

    trainer

    dataset_letters = ['ge3', 'siemens3', 'philips3']
    samples = list()
    dataset_probs = list()
    augmentation = list()
    for i in range(config['n_datasets']):
        samples.append(config['sample_' + dataset_letters[i]])
        dataset_probs.append(config['prob_' + dataset_letters[i]])
        augmentation.append(config['transform_' + dataset_letters[i]])

    train_loader_list, test_loader_list = get_all_data_loaders(config, config['n_datasets'], samples, augmentation, config['trim'])

    loader_sizes = list()

    for l in train_loader_list:

        loader_sizes.append(len(l))

    loader_sizes = np.asarray(loader_sizes)
    n_batches = loader_sizes.min()

    # Setup logger and output folders.
    # model_name = os.path.splitext(os.path.basename(opts.config))[0]
    output_directory = config['output_path'] + "models_codagan/codagan_" +  config['trainer']
    checkpoint_directory, image_directory = prepare_sub_folder(output_directory)
    # shutil.copy(opts.config, output_directory + '/config.yaml') # Copy config file to output folder.

    # Start training.
    epochs = config['max_iter']

    # if config['resume'] == -1:
    #     wandb_id = wandb.util.generate_id()
    #     wandb.init(project="CODAGAN", entity="autoda", resume="allow", id=wandb_id,
    #                 config={})
    # else:
    #     wandb_id = checkpoint["wandb_id"]
    #     wandb.init(project="CODAGAN", entity="autoda", resume="allow", id=wandb_id,
    #                     config={})

    for ep in range(max(config['resume'], 0), epochs):

        print('Start of epoch ' + str(ep + 1) + '...')

        trainer.update_learning_rate()

        print('    Training...')
        ctr = 0
        epoch_gen_loss = 0
        epoch_dis_loss = 0
        epoch_sup_loss = 0
        for it, data in enumerate(zip(*train_loader_list)):

            images_list = list()
            labels_list = list()
            use_list = list()

            for i in range(config['n_datasets']):

                images = data[i][0]
                labels = data[i][1]
                use = data[i][2].to(dtype=torch.uint8)

                images_list.append(images)
                labels_list.append(labels)
                use_list.append(use)

            # Randomly selecting datasets.
            perm = np.random.choice(config['n_datasets'], 2, replace=False, p=dataset_probs)
            print('        Ep: ' + str(ep + 1) + ', it: ' + str(it + 1) + '/' + str(n_batches) + ', domain pair: ' + str(perm))

            index_1 = perm[0]
            index_2 = perm[1]

            images_1 = images_list[index_1]
            images_2 = images_list[index_2]

            labels_1 = labels_list[index_1]
            labels_2 = labels_list[index_2]

            use_1 = use_list[index_1]
            use_2 = use_list[index_2]

            images_1, images_2 = Variable(images_1), Variable(images_2)

            # Main training code.
            if (ep + 1) <= int(0.75 * epochs):

                # If in Full Training mode.
                trainer.set_sup_trainable(True)
                trainer.set_gen_trainable(True)

                epoch_dis_loss += trainer.dis_update(images_1, images_2, index_1, index_2, config)
                epoch_gen_loss += trainer.gen_update(images_1, images_2, index_1, index_2, config)

            else:

                # If in Supervision Tuning mode.
                trainer.set_sup_trainable(True)
                trainer.set_gen_trainable(False)

            labels_1 = labels_1.to(dtype=torch.long)
            labels_1[labels_1 > 0] = 1
            labels_1 = Variable(labels_1, requires_grad=False)

            labels_2 = labels_2.to(dtype=torch.long)
            labels_2[labels_2 > 0] = 1
            labels_2 = Variable(labels_2, requires_grad=False)

            epoch_sup_loss += trainer.sup_update(images_1, images_2, labels_1, labels_2, index_1, index_2, use_1, use_2, config)
            ctr += 1

        wandb.log({'epoch': ep, 'Epoch Gen Loss': epoch_gen_loss/ctr, 'Epoch Dis Loss': epoch_dis_loss/ctr, 'Epoch Sup Loss': epoch_sup_loss/ctr})            
        print("Epoch: {}, Gen Loss = {}, Dis Loss = {}, Sup Loss = {}".format(ep, epoch_gen_loss/ctr, epoch_dis_loss/ctr, epoch_sup_loss/ctr))

        if (ep + 1) % config['snapshot_save_iter'] == 0:

            trainer.save(checkpoint_directory, (ep + 1))

        for i in range(config['n_datasets']):

            print('    Testing ' + dataset_letters[i] + '...')

            jacc_list = list()
            for it, data in enumerate(test_loader_list[i]):

                images = data[0]
                labels = data[1]
                use = data[2]
                path = data[3]

                images = Variable(images)

                labels = labels.to(dtype=torch.long)
                labels[labels > 0] = 1
                labels = Variable(labels, requires_grad=False)

                jacc, pred, iso = trainer.sup_forward(images, labels, 0, config)
                jacc_list.append(jacc)

                images_path = os.path.join(image_directory, 'originals', path[0])
                labels_path = os.path.join(image_directory, 'labels', path[0])
                pred_path = os.path.join(image_directory, 'predictions', path[0])

                np_images = images.cpu().numpy().squeeze()
                np_labels = labels.cpu().numpy().squeeze()

                io.imsave(images_path, norm(np_images, config['input_dim'] != 1))
                io.imsave(labels_path, norm(np_labels))
                io.imsave(pred_path, norm(pred))

            jaccard = np.asarray(jacc_list)

            print('        Test ' + dataset_letters[i] + ' Jaccard epoch ' + str(ep + 1) + ': ' + str(100 * jaccard.mean()) + ' +/- ' + str(100 * jaccard.std()))
            wandb.log({'epoch': ep, 'Jaccard': jaccard.mean()})            
