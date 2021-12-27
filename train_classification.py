from __future__ import print_function
from tensorboardX import SummaryWriter  
from IPython.core.debugger import set_trace
from datetime import datetime
import os
import shutil
import argparse
import time
import json
import glob
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score
from datasets import create_datasets
import yaml
from easydict import EasyDict as edict
import pandas as pd
import numpy as np
from collections import defaultdict
from matplotlib import pyplot as plt
from tqdm import tqdm
from joblib import Parallel, delayed
import torch
from torch import nn
from torch import autograd
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchio as tio

from utils import check_patch, pad_arrays, normalize, load, create_dicts, trim, video, video_comparison
from IPython.core.display import display, HTML
from train import DiceScoreBinary, parse_args, save

from multiprocessing import cpu_count
N_CPU = cpu_count()
SEED = 42


def one_epoch(model, 
                criterion, 
                opt, 
                config, 
                dataloader, 
                device, 
                writer, 
                epoch, 
                metric_dict_epoch, 
                n_iters_total=0,
                augmentation=None, 
                is_train=True):

    phase_name = 'train' if is_train else 'val'
    loss_name = config.opt.criterion
    metric_dict = defaultdict(list)
    patch_size = config.dataset.patch_size
    patch_batch_size = config.dataset.patch_batch_size
    queue_length = config.dataset.queue_length
    samples_per_volume = config.dataset.samples_per_volume
    negative_sampling_prob = config.dataset.negative_sampling_prob
    classification_threshold = config.model.classification_threshold

    # used to turn on/off gradients
    grad_context = torch.autograd.enable_grad if is_train else torch.no_grad
    with grad_context():
        iterator = enumerate(dataloader)

        for iter_i, (brain_tensor, label_tensor) in iterator:
            

            #####################
            # SETUP DATALOADERS #
            #####################
            # brain_tensor - [1,C,H,W,D]
            # label_tensor - [1,1,H,W,D]
            sampling_map = label_tensor[0].clone() # leave only channel dim
            sampling_map[sampling_map == 0] = negative_sampling_prob
            subject = tio.Subject(t1=tio.ScalarImage(tensor=brain_tensor[0]),
                                  label=tio.LabelMap(tensor=label_tensor[0]),
                                  sampling_map=sampling_map)

            if is_train:
                if augmentation is not None:
                    subject = augmentation(subject)

                subjects_dataset = tio.SubjectsDataset([subject])
                sampler = tio.data.WeightedSampler(patch_size, 'sampling_map')
                
                patches_queue = tio.Queue(
                    subjects_dataset,
                    queue_length,
                    samples_per_volume,
                    sampler,
                    num_workers=4
                )

                patch_loader = DataLoader(
                    patches_queue,
                    batch_size=patch_batch_size,
                    num_workers=1,  # this must be 0
                )

            else:
                patch_overlap = patch_size//2 
                grid_sampler = tio.inference.GridSampler(
                    subject, # validation subject
                    patch_size,
                    patch_overlap,
                )

                patch_loader = torch.utils.data.DataLoader(grid_sampler, batch_size=patch_batch_size)
                aggregator = tio.inference.GridAggregator(grid_sampler, overlap_mode='average')
                # aggregator_topk = tio.inference.GridAggregator(grid_sampler)

            ########################
            # ITERATE OVER PATCHES #
            #############################################################################
            N_fcd = label_tensor.sum() # number of FCD pixels in target
            # number of FCD pixels in patch to be considered as FCD patch
            PATCH_FCD_THRESHOLD = config.dataset.patch_fcd_threshold 
            metric_dict_patch = defaultdict(list)
            for patches_batch in patch_loader:
                
                inputs = patches_batch['t1'][tio.DATA].to(device)  # [bs,C,p,p,p]
                targets = patches_batch['label'][tio.DATA].to(device) # [bs,1,p,p,p]

                # set_trace()

                logits = model(inputs)

                # set_trace()
                
                targets_ = targets.sum([-1,-2,-3]) / N_fcd > PATCH_FCD_THRESHOLD
                targets_ = targets_.type(torch.float32)

                loss = criterion(logits, targets_) # [bs,1], [bs,1]
                
                if is_train:
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                else:
                    locations = patches_batch[tio.LOCATION]
                    outputs = torch.ones_like(targets)*torch.sigmoid(logits)[...,None,None,None] # [bs,1,p,p,p]
                    aggregator.add_batch(outputs, locations)

                #####################
                # per-PATCH METRICS #
                #####################
                # map to and remove last dim
                logits_prob_np = torch.sigmoid(logits.squeeze(-1)).detach().cpu().numpy()
                targets_np = targets_.squeeze(-1).detach().cpu().numpy().astype(int)

                targets_pred_np = (logits_prob_np > classification_threshold).astype(int)
                # precision = precision_score(targets_np, targets_pred_np, zero_division=0)
                # recall = recall_score(targets_np, targets_pred_np, zero_division=0)
                accuracy = accuracy_score(targets_np, targets_pred_np)

                # metric_dict_patch['precision'].append(precision)
                # metric_dict_patch['recall'].append(recall)
                metric_dict_patch['accuracy'].append(accuracy)
                metric_dict_patch[f'{loss_name}'].append(loss.item())

            ##############################################################################

            for k,v in metric_dict_patch.items():
                metric_dict[k].append(np.mean(v))

            if not is_train:
                
                output_tensor = aggregator.get_output_tensor().unsqueeze(1) # [1,1,H,W,D]
                # output_tensor = output_tensor / output_tensor.max()
                # set_trace()
                dice = DiceScoreBinary(output_tensor, label_tensor).item()  
                metric_dict['dice_score'].append(dice)

            if is_train and writer is not None:
                for title, value in metric_dict.items():
                    writer.add_scalar(f"{phase_name}_{title}", value[-1], n_iters_total)
            
            # set_trace()
            ############
            # PRINTING #
            ############
            loss_mean = metric_dict[f'{loss_name}'][-1]
            message = f'Epoch:{epoch}, iter:{iter_i}, loss-{loss_name}:{loss_mean}'
            if not is_train:
                message += f', DICE:{dice}'
            print(message)

            n_iters_total += 1

    try:
        target_metric = 0
        for title, value in metric_dict.items():
            m = np.mean(value)
            if writer is not None:
                writer.add_scalar(f"{phase_name}_{title}_epoch", m, epoch)
            metric_dict_epoch[phase_name + '_' + title].append(m)
            if 'dice_score' == title:
                target_metric = m

    except Exception as e:
        print ('Exception:', str(e), 'Failed to save writer')

    return n_iters_total, target_metric

def main(args):


    print(f'Available devices: {torch.cuda.device_count()}')

    with open(args.config) as fin:
        config = edict(yaml.safe_load(fin))

    # setting logs
    MAKE_LOGS = config.make_logs
    SAVE_MODEL = config.opt.save_model if hasattr(config.opt, "save_model") else True
    DEVICE = config.opt.device if hasattr(config.opt, "device") else 1
    device = torch.device(DEVICE)
    assert config.opt.train_batch_size == 1 and config.opt.val_batch_size == 1

    experiment_name = '{}@{}'.format(args.experiment_comment, datetime.now().strftime("%d.%m.%Y-%H:%M:%S"))
    print("Experiment name: {}".format(experiment_name))
    
    if MAKE_LOGS:
        experiment_dir = os.path.join(args.logdir, experiment_name)
        if os.path.isdir(experiment_dir):
            shutil.rmtree(experiment_dir)
        os.makedirs(experiment_dir)
        shutil.copy(args.config, os.path.join(experiment_dir, "config.yaml"))
    writer = SummaryWriter(os.path.join(experiment_dir, "tb")) if MAKE_LOGS else None

    ################
    # CREATE MODEL #
    ################
    model = torchvision.models.video.r2plus1d_18(pretrained=False, progress=False) 
    conv3d_1 = model.stem[0]
    input_channels = 1 # MRI brain itself
    if config.dataset.use_features:
        input_channels += len(config.dataset.features) if hasattr(config.dataset,'features') else 0
    model.stem[0] = nn.Conv3d(in_channels=input_channels,
                             out_channels=conv3d_1.out_channels,
                             kernel_size=conv3d_1.kernel_size,
                             padding=conv3d_1.padding,
                             bias=conv3d_1.bias)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model = model.to(device)
    print('Model created!')

    if hasattr(config.model, 'weights'):
        model_dict = torch.load(os.path.join(config.model.weights, 'checkpoints/weights.pth'))
        print(f'LOADING from {config.model.weights} \n epoch:', model_dict['epoch'])
        model.load_state_dict(model_dict['model_state'])

    ###################
    # CREATE DATASETS #
    ###################
    train_dataset, val_dataset = create_datasets(config)

    train_dataloader = DataLoader(train_dataset, batch_size=config.opt.train_batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config.opt.val_batch_size, shuffle=False)
    print(len(train_dataloader), len(val_dataloader))

    augmentation = None
    if config.opt.augmentation:
        symmetry = tio.RandomFlip(axes=0)
        bias = tio.RandomBiasField(coefficients=0.3)
        noise = tio.RandomNoise(std=(0,1e-3))
        affine = tio.RandomAffine(scales=(0.9, 1.1, 0.9, 1.1, 0.9, 1.1), 
                                 degrees=5,
                                 translation=(1,1,1),
                                 center='image',
                                 default_pad_value=0)
        rescale = tio.RescaleIntensity(out_min_max=(0, 1))
        augmentation = tio.Compose([symmetry, bias, noise, affine, rescale])

    ################
    # CREATE OPTIM #
    ################
    criterion = {
        "BCE": nn.BCEWithLogitsLoss(), # [logits:float32, target:float32]
    }[config.opt.criterion]
    opt = optim.Adam(model.parameters(), lr=config.opt.lr)


    ############
    # TRAINING #
    ############
    print('Start training!')
    metric_dict_epoch = defaultdict(list)
    n_iters_total_train = 0 
    n_iters_total_val = 0
    target_metric = 0
    target_metric_prev = -1
    epoch_to_save = 0
    try:
        for epoch in range(config.opt.start_epoch, config.opt.n_epochs):
            print (f'TRAIN EPOCH: {epoch} ... ')
            n_iters_total_train, _  = one_epoch(model, 
                                            criterion, 
                                            opt, 
                                            config, 
                                            train_dataloader, 
                                            device, 
                                            writer, 
                                            epoch, 
                                            metric_dict_epoch, 
                                            n_iters_total_train,
                                            augmentation=augmentation,
                                            is_train=True)

            print (f'VAL EPOCH: {epoch} ... ')
            n_iters_total_val, target_metric = one_epoch(model, 
                                                criterion, 
                                                opt, 
                                                config, 
                                                val_dataloader, 
                                                device, 
                                                writer, 
                                                epoch, 
                                                metric_dict_epoch, 
                                                n_iters_total_val,
                                                augmentation=None,
                                                is_train=False)

            if SAVE_MODEL and MAKE_LOGS:
                if target_metric > target_metric_prev:
                    save(experiment_dir, model, opt, epoch)
                    target_metric_prev = target_metric

    except:
        # keyboard interrupt
        if MAKE_LOGS:
            np.save(os.path.join(experiment_dir, 'metric_dict_epoch'), metric_dict_epoch)

if __name__ == '__main__':
    args = parse_args()
    main(args)
