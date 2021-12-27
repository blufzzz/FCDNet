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
from utils import trim, normalize, normalize_
from sklearn.metrics import accuracy_score, roc_auc_score
from models.v2v import V2VModel
import nibabel
import yaml
from easydict import EasyDict as edict


def create_datasets(config):

    if config.dataset.dataset_type == 'interpolated': 
        train_dataset = BrainMaskDataset(config.dataset, train=True)
        val_dataset = BrainMaskDataset(config.dataset, train=False)
    elif config.dataset.dataset_type == 'brats_interpolated': 
        train_dataset = Brats2020BrainMaskDataset(config.dataset, train=True)
        val_dataset = Brats2020BrainMaskDataset(config.dataset, train=False)
    elif config.dataset.dataset_type == 'patches':
        train_dataset = BrainMaskDataset(config.dataset, train=True)
        val_dataset = BrainMaskDataset(config.dataset, train=False)
    else:
        raise RuntimeError('Wrond `dataset_type`!')
    return train_dataset, val_dataset



class Brats2020BrainMaskDataset(Dataset):

    def __init__(self, config, train=True):
        self.root = config.root
        self.train = train
        self.trim_background = config.trim_background

        self.metadata = np.load('metadata_brats2020.npy',allow_pickle=True).item()
        metadata_key = 'train' if self.train else 'test'
        self.labels = self.metadata[metadata_key]

        self.paths = [os.path.join(self.root, f'BraTS20_Training_{k}/BraTS20_Training_{k}') for k in self.labels]
        self.use_features = False
        self.features = None

    def __getitem__(self, idx):

        path = self.paths[idx]
        
        brain = nibabel.load(path + '_t2.nii.gz').get_fdata()
        label = nibabel.load(path + '_seg.nii.gz').get_fdata()
        label = (label > 0).astype(int)
        brain = normalize(brain)

        brain_tensor_torch = torch.tensor(brain, dtype=torch.float32).unsqueeze(0)
        label_tensor_torch = torch.tensor(label, dtype=torch.float32)


        if self.trim_background:
            brain_tensor_torch, label_tensor_torch, mask_tensor_torch = trim(brain_tensor_torch, 
                                                                             label_tensor_torch)
            

        return brain_tensor_torch, label_tensor_torch.unsqueeze(0)

    def __len__(self):
        return len(self.paths)


class BrainMaskDataset(Dataset):


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
            brain_tensor_torch, label_tensor_torch, mask_tensor_torch = trim(brain_tensor_torch, 
                                                                                label_tensor_torch,
                                                                                mask_tensor_torch)

        return brain_tensor_torch,\
                label_tensor_torch.unsqueeze(0)

    def __len__(self):
        return len(self.paths)
