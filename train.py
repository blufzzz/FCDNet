from tensorboardX import SummaryWriter  
from IPython.core.debugger import set_trace
from datetime import datetime
import os
import shutil
import argparse
import time
import json
from collections import defaultdict
import pickle
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch import autograd
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.parallel import DistributedDataParallel

from models.v2v import V2VModel

import yaml
from easydict import EasyDict as edict


def save(experiment_dir, model, opt, epoch):

    checkpoint_dir = os.path.join(experiment_dir, "checkpoints") # , "{:04}".format(epoch)
    os.makedirs(checkpoint_dir, exist_ok=True)

    dict_to_save = {'model_state': model.state_dict(),'opt_state' : opt.state_dict()}

    torch.save(dict_to_save, os.path.join(checkpoint_dir, "weights.pth"))


def DiceLoss(input, target):
    '''
    Binary Dice loss
    target - binary mask [bs,1,ps,ps,ps]
    input - [bs,2,ps,ps,ps]
    '''

    # create "background" class

    target_float = target.unsqueeze(1).type(input.dtype)
    background_tensor = torch.abs(target_float- 1)
    target_stacked = torch.cat([background_tensor, target_float], dim=1) # [bs,2,ps,ps,ps]

    intersection = torch.sum(input * target_stacked, dim=(1,2,3,4)) # [bs,]
    cardinality = torch.sum(input + target_stacked, dim=(1,2,3,4)) # [bs,]

    dice_score = 2. * intersection / (cardinality + 1e-7)
    return torch.mean(1. - dice_score)


class CatBrainMaskPatchLoader(Dataset):
    
    def __init__(self, config, train=True):
        self.root = config.root
        self.train = train
        self.patch_size = config.patch_size
        self.use_features = config.use_features

        which_metadata = 'train' if train else 'test' 
        self.metadata = pd.read_csv(os.path.join(self.root, f'metadata_{which_metadata}'))

    def __getitem__(self, idx):
        
        metaindex = self.metadata.iloc[idx]
        
        tensor = torch.load(os.path.join(self.root, f'tensor_{metaindex.label}'))
        x,y,z = metaindex[['x','y','z']].astype(int)

        x1,x2 = x-self.patch_size//2, x+self.patch_size//2
        y1,y2 = y-self.patch_size//2, y+self.patch_size//2
        z1,z2 = z-self.patch_size//2, z+self.patch_size//2

        if self.use_features:
            brain_patch = tensor[0:-1,x1:x2,y1:y2,z1:z2] # brain 
        else:
            brain_patch = tensor[0:1,x1:x2,y1:y2,z1:z2] # brain 
        label_patch = tensor[-1,x1:x2,y1:y2,z1:z2] # label
    
        return brain_patch, label_patch

    def __len__(self):
        return self.metadata.shape[0]


class CatBrainMaskLoader(Dataset):

    def __init__(self, config, train=True):
        raise RuntimeError

    # def __init__(self, config, train=True):
    #     self.root = config.root
    #     self.train = train
    #     self.data_path = os.path.join(self.root, 'train' if self.train else 'test')
    #     self.paths = [os.path.join(data_path, p) for p in os.listdir(data_path)]

    # def __getitem__(self, idx):
    #     brain_tensor_torch, mask_tensor_torch = torch.load(self.paths[idx])
    #     return brain_tensor_torch, mask_tensor_torch

    # def __len__(self):
    #     return len(self.paths)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--logdir", type=str, default="")
    parser.add_argument('--experiment_comment', default='', type=str)
    args = parser.parse_args()
    return args


def create_datesets(config):

    if config.dataset.dataset_type == 'whole_brain': 
        train_dataset = CatBrainMaskLoader(config.dataset, train=True)
        val_dataset = CatBrainMaskLoader(config.dataset, train=True)
    elif config.dataset.dataset_type == 'patches':
        train_dataset = CatBrainMaskPatchLoader(config.dataset, train=True)
        val_dataset = CatBrainMaskPatchLoader(config.dataset, train=False)
    else:
        raise RuntimeError('Wrond `dataset_type`!')

    return train_dataset, val_dataset


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
                is_train=True):

    phase_name = 'train' if is_train else 'val'
    metric_dict = defaultdict(list)
    # used to turn on/off gradients
    grad_context = torch.autograd.enable_grad if is_train else torch.no_grad
    with grad_context():
        iterator = enumerate(dataloader)

        for iter_i, (brain_tensor, mask_tensor) in iterator:

            # prepare
            brain_tensor = brain_tensor.to(device) # [bs,1,ps,ps,ps]
            mask_tensor = mask_tensor.type(torch.long).to(device) # [bs,ps,ps,ps]

            if config.interpolate:
                brain_tensor = F.interpolate(brain_tensor, config.interpolation_size)
                mask_tensor = F.interpolate(mask_tensor, config.interpolation_size)


            # forward pass
            mask_tensor_predicted = model(brain_tensor) # [bs,2,ps,ps,ps]
            ce_loss = criterion(mask_tensor_predicted, mask_tensor)


            if is_train:
                opt.zero_grad()
                ce_loss.backward()
                opt.step()

            metric_dict['ce_loss'].append(ce_loss.item())

            # Dice loss
            dice_loss = DiceLoss(mask_tensor_predicted, mask_tensor)
            metric_dict['dice_loss'].append(dice_loss.item())

            print(f'Epoch: {epoch}, iter: {iter_i}, ce_loss: {ce_loss.item()}, dice_loss: {dice_loss.item()}')

            if is_train and writer is not None:
                for title, value in metric_dict.items():
                    writer.add_scalar(f"{phase_name}_{title}", value[-1], n_iters_total)

            n_iters_total += 1


    try:
        for title, value in metric_dict.items():
            m = np.mean(value)
            if writer is not None:
                writer.add_scalar(f"{phase_name}_{title}_epoch", m, epoch)
            if not config.silent:
                print(f'Epoch value: {phase_name}_{title}= {m}')
            metric_dict_epoch[phase_name + '_' + title].append(m)

    except Exception as e:
        print ('Exception:', str(e), 'Failed to save writer')


    return n_iters_total

def main(args):


    print(f'Available devices: {torch.cuda.device_count()}')
    

    with open(args.config) as fin:
        config = edict(yaml.safe_load(fin))

    # setting logs
    MAKE_LOGS = config.make_logs
    SAVE_MODEL = config.opt.save_model if hasattr(config.opt, "save_model") else True
    DEVICE = config.opt.device if hasattr(config.opt, "device") else 1
    device = torch.device(DEVICE)
    experiment_name = '{}@{}'.format(args.experiment_comment, datetime.now().strftime("%d.%m.%Y-%H:%M:%S"))
    print("Experiment name: {}".format(experiment_name))
    
    if MAKE_LOGS:
        experiment_dir = os.path.join(args.logdir, experiment_name)
        if os.path.isdir(experiment_dir):
            shutil.rmtree(experiment_dir)
        os.makedirs(experiment_dir)

    writer = SummaryWriter(os.path.join(experiment_dir, "tb")) if MAKE_LOGS else None
    model = V2VModel(config).to(device)
    print('Model created!')

    # setting datasets
    train_dataset, val_dataset = create_datesets(config)
    train_dataloader = DataLoader(train_dataset, batch_size=config.opt.train_batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config.opt.val_batch_size, shuffle=True)

    # setting model stuff
    criterion = {
        "CE": torch.nn.CrossEntropyLoss(),
        "FDice": None
    }[config.opt.criterion]

    opt = optim.Adam(model.parameters(), lr=config.opt.lr)

    # training
    print('Start training!')

    metric_dict_epoch = defaultdict(list)
    n_iters_total_train = 0
    n_iters_total_val = 0
    try:
        for epoch in range(config.opt.start_epoch, config.opt.n_epochs):
            print (f'TRAIN EPOCH: {epoch} ... ')
            n_iters_total_train = one_epoch(model, 
                                            criterion, 
                                            opt, 
                                            config, 
                                            train_dataloader, 
                                            device, 
                                            writer, 
                                            epoch, 
                                            metric_dict_epoch, 
                                            n_iters_total_train,
                                            is_train=True)

            print (f'VAL EPOCH: {epoch} ... ')
            n_iters_total_val = one_epoch(model, 
                                            criterion, 
                                            opt, 
                                            config, 
                                            val_dataloader, 
                                            device, 
                                            writer, 
                                            epoch, 
                                            metric_dict_epoch, 
                                            n_iters_total_val,
                                            is_train=False)

            if SAVE_MODEL:
                save(experiment_dir, model, opt, epoch)
    except:
        # keyboard interrupt
        if MAKE_LOGS:
            np.save(os.path.join(experiment_dir, 'metric_dict_epoch'), metric_dict_epoch)

if __name__ == '__main__':
    args = parse_args()
    print("args: {}".format(args))
    main(args)