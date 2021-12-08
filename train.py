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
from utils import trim, normalized
from sklearn.metrics import accuracy_score, roc_auc_score
from models.v2v import V2VModel
import torchio as tio

import yaml
from easydict import EasyDict as edict

from losses import focal_tversky_loss


def save(experiment_dir, model, opt, epoch):

    checkpoint_dir = os.path.join(experiment_dir, "checkpoints") # , "{:04}".format(epoch)
    os.makedirs(checkpoint_dir, exist_ok=True)

    dict_to_save = {'model_state': model.state_dict(),'opt_state' : opt.state_dict(), 'epoch':epoch}

    torch.save(dict_to_save, os.path.join(checkpoint_dir, "weights.pth"))


def DiceScoreBinary(input, 
                    target, 
                    include_backgroud=False, 
                    weights=None):
    '''
    Binary Dice score
    target - binary mask [bs,1,ps,ps,ps], 1 for foreground, 0 for background
    input - [bs,1,ps,ps,ps], probability [0,1]
    '''

    # create "background" class
    
    if include_backgroud:
        target_float = target.type(input.dtype)
        background_tensor = torch.abs(target_float - 1.)
        target_stacked = torch.cat([background_tensor, target_float], dim=1) # [bs,2,ps,ps,ps]
        input_stacked = torch.cat([1-input, input], dim=1) # [bs,2,ps,ps,ps]
        
        intersection = torch.sum(input_stacked * target_stacked, dim=(2,3,4)) # [bs,2]
        cardinality = torch.sum(torch.pow(input_stacked,2) + torch.pow(target_stacked,2), dim=(2,3,4)) # [bs,2]
        dice_score = 2. * intersection / (cardinality + 1e-7) # [bs,2]
        
        if weights is not None:
            dice_score = (dice_score*weights).sum(1)
        
    else:

        target = target.squeeze(1) # cast to float and squeeze channel # .type(input.dtype)
        input = input.squeeze(1) # squeeze channel
        
        intersection = torch.sum(input * target, dim=(1,2,3)) # [bs,]
        cardinality = torch.sum(torch.pow(input,2) + torch.pow(target,2), dim=(1,2,3)) # [bs,]
        dice_score = 2. * intersection / (cardinality + 1e-7)

    return dice_score.mean()


def DiceLossBinary(*args, **kwargs):
    return 1 - DiceScoreBinary(*args, **kwargs)



class CatBrainMaskLoader_npy(Dataset):

    def __init__(self, config, train=True):
        raise RuntimeError

    def __init__(self, config, train=True):

        self.root = config.root
        self.train = train
        self.trim_background = config.trim_background

        self.metadata = np.load('metadata.npy',allow_pickle=True).item()
        metadata_key = 'train' if self.train else 'test'
        self.labels = self.metadata[metadata_key]

        for label in self.labels:
            if not os.path.isfile(os.path.join(self.root, 'labels', f'{label}.npy')):
                self.labels.remove(label)
                print(f'LABEL: {label} removed!')

        self.paths_brains = [os.path.join(self.root, 't1_brains', f'{k}.npy') for k in self.labels]
        self.paths_labels = [os.path.join(self.root, 'labels', f'{k}.npy') for k in self.labels]

    def trim_npy(self, brain_tensor, label_tensor):
    
        '''
        mask_tensor - [H,W,D]
        brain_tensor - [H,W,D]
        label_tensor - [H,W,D]
        
        '''
        X,Y,Z = brain_tensor.shape
        
        background = brain_tensor[0,0,0]
        brain_tensor_ = brain_tensor != background 
        
        x1,x2 = np.arange(X)[np.sum(brain_tensor_, axis=(1,2)) > 0][[0,-1]]
        y1,y2 = np.arange(Y)[np.sum(brain_tensor_,axis=(0,2)) > 0][[0,-1]]
        z1,z2 = np.arange(Z)[np.sum(brain_tensor_,axis=(0,1)) > 0][[0,-1]]
        
        brain_tensor_trim = brain_tensor[x1:x2][:,y1:y2][:,:,z1:z2]
        label_tensor_trim = label_tensor[x1:x2][:,y1:y2][:,:,z1:z2]
        
        return brain_tensor_trim, label_tensor_trim

    def __getitem__(self, idx):


        brain = np.load(self.paths_brains[idx])
        background = brain[0,0,0]
        mask = brain != background
        brain[mask] = normalized(brain[mask])
        label = np.load(self.paths_labels[idx])

        if self.trim_background:
            brain, label = self.trim_npy(brain, label)

        brain_tensor_torch = torch.tensor(brain,  dtype=torch.float32)
        label_tensor_torch = torch.tensor(label,  dtype=torch.float32)

        return brain_tensor_torch.unsqueeze(0), label_tensor_torch.unsqueeze(0)

    def __len__(self):
        return len(self.labels)




class CatBrainMaskLoader(Dataset):

    def __init__(self, config, train=True):
        raise RuntimeError

    def __init__(self, config, train=True):
        self.root = config.root
        self.train = train
        self.trim_background = config.trim_background

        self.metadata = np.load('metadata.npy',allow_pickle=True).item()
        metadata_key = 'train' if self.train else 'test'
        self.labels = self.metadata[metadata_key]

        self.paths = [os.path.join(self.root, f'tensor_{k}') for k in self.labels]
        self.use_features = config.use_features
        self.features = config.features if hasattr(config, 'features') else None

    def __getitem__(self, idx):

        tensor_dict = torch.load(self.paths[idx])
        brain_tensor_torch = tensor_dict['brain']
        mask_tensor_torch = tensor_dict['mask']
        label_tensor_torch = tensor_dict['label']

        if self.use_features:
            assert self.features is not None
            brain_tensor_torch = torch.stack([brain_tensor_torch] + \
                                            [tensor_dict[f] for f in self.features],
                                            dim=0)
        else:
            brain_tensor_torch = brain_tensor_torch.unsqueeze(0)

        if self.trim_background:
            brain_tensor_torch, mask_tensor_torch, label_tensor_torch = trim(brain_tensor_torch, 
                                                                                mask_tensor_torch, 
                                                                                label_tensor_torch)

        return brain_tensor_torch, label_tensor_torch.unsqueeze(0)

    def __len__(self):
        return len(self.paths)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--logdir", type=str, default="")
    parser.add_argument('--experiment_comment', default='', type=str)
    args = parser.parse_args()
    return args


def create_datesets(config):

    if config.dataset.dataset_type == 'interpolated': 
        train_dataset = CatBrainMaskLoader(config.dataset, train=True)
        val_dataset = CatBrainMaskLoader(config.dataset, train=False)
    elif config.dataset.dataset_type == 'patches':
        train_dataset = CatBrainMaskPatchLoader(config.dataset, train=True)
        val_dataset = CatBrainMaskPatchLoader(config.dataset, train=False)

    elif config.dataset.dataset_type == 'interpolated_npy':
        train_dataset = CatBrainMaskLoader_npy(config.dataset, train=True)
        val_dataset = CatBrainMaskLoader_npy(config.dataset, train=False)

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
                augmentation=None, 
                is_train=True):

    phase_name = 'train' if is_train else 'val'
    loss_name = config.opt.criterion
    metric_dict = defaultdict(list)
    # used to turn on/off gradients
    grad_context = torch.autograd.enable_grad if is_train else torch.no_grad
    with grad_context():
        iterator = enumerate(dataloader)

        for iter_i, (brain_tensor, label_tensor) in iterator:
            
            # set_trace()

            if (augmentation is not None) and is_train:
                assert config.opt.train_batch_size == 1

                subject = tio.Subject(
                    t1=tio.ScalarImage(tensor=brain_tensor[0]),
                    label=tio.LabelMap(tensor=label_tensor[0]),
                    diagnosis='positive'
                )

                transformed = augmentation(subject)
                brain_tensor = transformed['t1'].tensor.unsqueeze(0)
                label_tensor = transformed['label'].tensor.unsqueeze(0)

                # set_trace()

            if config.interpolate:
                brain_tensor = F.interpolate(brain_tensor, config.interpolation_size).to(device)
                label_tensor = F.interpolate(label_tensor, config.interpolation_size).to(device) # unsqueeze channel
            else:
                brain_tensor = brain_tensor.to(device)
                label_tensor = label_tensor.to(device)

            # forward pass
            label_tensor_predicted = model(brain_tensor) # [bs,1,ps,ps,ps]
            
            # set_trace()
            loss = criterion(label_tensor_predicted, label_tensor)

            if is_train:
                opt.zero_grad()
                loss.backward()
                opt.step()

            metric_dict[f'{loss_name}'].append(loss.item())

            dice_score = DiceScoreBinary(label_tensor_predicted, label_tensor)
            metric_dict['dice_score'].append(dice_score.item())

            print(f'Epoch: {epoch}, iter: {iter_i}, Loss_{loss_name}: {loss.item()}, DICE: {dice_score.item()}')

            if is_train and writer is not None:
                for title, value in metric_dict.items():
                    writer.add_scalar(f"{phase_name}_{title}", value[-1], n_iters_total)

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

    # os.system(f'CUDA_VISIBLE_DEVICES={DEVICE}')

    experiment_name = '{}@{}'.format(args.experiment_comment, datetime.now().strftime("%d.%m.%Y-%H:%M:%S"))
    print("Experiment name: {}".format(experiment_name))
    
    if MAKE_LOGS:
        experiment_dir = os.path.join(args.logdir, experiment_name)
        if os.path.isdir(experiment_dir):
            shutil.rmtree(experiment_dir)
        os.makedirs(experiment_dir)

        shutil.copy(args.config, os.path.join(experiment_dir, "config.yaml"))

    writer = SummaryWriter(os.path.join(experiment_dir, "tb")) if MAKE_LOGS else None
    model = V2VModel(config).to(device)
    print('Model created!')

    # setting datasets
    train_dataset, val_dataset = create_datesets(config)
    train_dataloader = DataLoader(train_dataset, batch_size=config.opt.train_batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config.opt.val_batch_size, shuffle=False)
    print(len(train_dataloader), len(val_dataloader))


    # if 
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
        transform = tio.Compose([symmetry, bias, noise, affine, rescale]) # , affine


    # setting model stuff
    criterion = {
        # "CE": torch.nn.CrossEntropyLoss(),
        "BCE": torch.nn.BCELoss(), # [probabilities, target]
        "Dice": DiceLossBinary,
        "FocalTversky":focal_tversky_loss(),
        
    }[config.opt.criterion]

    opt = optim.Adam(model.parameters(), lr=config.opt.lr)

    # training
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
                                            augmentation=transform,
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
    print("args: {}".format(args))
    main(args)
