from IPython.core.debugger import set_trace
import os
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from utils import trim, normalize
from torchsample import StratifiedSampler



def create_datasets(config):

    if config.dataset.dataset_type == 'fcd': 
        train_dataset = BrainMaskDataset(config.dataset, train=True)
        val_dataset = BrainMaskDataset(config.dataset, train=False)
    elif config.dataset.dataset_type == 'brats': 
        train_dataset = Brats2020Dataset(config.dataset, train=True)
        val_dataset = Brats2020Dataset(config.dataset, train=False)
    else:
        raise RuntimeError('Wrond `dataset_type`!')
    return train_dataset, val_dataset


class BrainPatchesDataset(Dataset):

    def __init__(self, brain, mask, label, coords, target, patch_size):
        
        self.brain = brain
        self.mask = mask
        self.label = label
        self.coords  = coords
        self.target = target
        self.patch_size = patch_size
        
    def __getitem__(self, idx):

        x,y,z = self.coords[idx]
        
        pd = self.patch_size//2
        x1,x2 = x-pd,x+pd 
        y1,y2 = y-pd,y+pd
        z1,z2 = z-pd,z+pd
        
        patch = self.brain[:,x1:x2,y1:y2,z1:z2]
        target = self.label[:,x1:x2,y1:y2,z1:z2]
        
        return patch, target

    def __len__(self):
        return len(self.target)



def BalancedSampler(subject, patch_size, patches_per_brain, patch_batch_size, label_ratio):

    
    brain, mask, label = subject['t1'].tensor, subject['mask'].tensor, subject['label'].tensor

    ps = (patch_size//2) + 1
    padding = (ps,ps, ps,ps, ps,ps, 0,0)
    brain = F.pad(brain, padding, "constant", 0)
    mask = F.pad(mask, padding, "constant", 0)
    label = F.pad(label, padding, "constant", 0)

    X,Y,Z = label.shape[-3:]

    neg_sample_mask = ((label[0] == 0)*mask[0]).type(torch.bool)
    pos_sample_mask = (label[0]).type(torch.bool)

    xyz_grid = torch.tensor(np.stack(np.meshgrid(np.arange(X), np.arange(Y), np.arange(Z), indexing='ij'), -1))

    neg_coords = xyz_grid[neg_sample_mask]
    pos_coords = xyz_grid[pos_sample_mask]

    N_neg = len(neg_coords)
    N_pos = len(pos_coords)

    neg_index = np.arange(N_neg)
    pos_index = np.arange(N_pos)

    N_neg_subsample = int((1-label_ratio)*patches_per_brain)
    N_pos_subsample = int(label_ratio*patches_per_brain)

    # random balanced subsample
    neg_index_subsample = np.random.choice(neg_index, size=N_neg_subsample)
    pos_index_subsample = np.random.choice(pos_index, size=N_pos_subsample)

    neg_coords_subsample = neg_coords[neg_index_subsample]
    pos_coords_subsample = pos_coords[pos_index_subsample]

    y = torch.cat([torch.zeros(N_neg_subsample), torch.ones(N_pos_subsample)])
    X = torch.cat([neg_coords_subsample, pos_coords_subsample])


    bp_dataset = BrainPatchesDataset(brain, mask, label, X, y, patch_size)
    sampler = StratifiedSampler(y, patch_batch_size)

    loader = DataLoader(bp_dataset, 
                              batch_size=patch_batch_size,
                              shuffle=False, 
                              sampler=sampler, 
                              num_workers=4)

    return loader


class Brats2020Dataset(Dataset):

    def __init__(self, config, train=True):
        self.root = config.root
        self.train = train
        self.trim_background = config.trim_background
        self.features = ['t1', 't1ce', 't2', 'flair']
        self.metadata_path = config.metadata_path

        self.metadata = np.load(self.metadata_path, allow_pickle=True).item()
        metadata_key = 'train' if self.train else 'test'
        self.labels = self.metadata[metadata_key]
        self.paths = [os.path.join(self.root, f'tensor_{k}') for k in self.labels]#[:1]
        self.features = config.features

    def __getitem__(self, idx):

        tensor_dict = torch.load(self.paths[idx]) # already normalized tensors
        label_tensor_torch = tensor_dict['seg']
        if 'mask' in tensor_dict.keys():
            mask_tensor_torch = tensor_dict['mask']
        else:
            mask_tensor_torch = None

        # features = sorted(set(self.features) - {'seg', 'mask'})

        brain_tensor_torch = torch.stack([tensor_dict[f] for f in self.features], dim=0) 

        if self.trim_background:
            brain_tensor_torch, label_tensor_torch, mask_tensor_torch = trim(brain_tensor_torch, 
                                                                             label_tensor_torch,
                                                                              mask_tensor_torch)
            
        return brain_tensor_torch,\
                mask_tensor_torch.unsqueeze(0),\
                label_tensor_torch.unsqueeze(0)

    def __len__(self):
        return len(self.paths) 


def add_xyz(input, mask):

    '''
    input - [C,H,W,D]
    '''

    _,X,Y,Z = input.shape

    xyz_grid = torch.tensor(np.stack(np.meshgrid(np.arange(X)/X, 
                            np.arange(Y)/Y, 
                            np.arange(Z)/Z, 
                            indexing='ij'), 0), dtype=input.dtype).to(input.device)

    return torch.cat([xyz_grid,input],0) * mask


class BrainMaskDataset(Dataset):


    def __init__(self, config, train=True):
        self.root = config.root
        self.train = train
        self.trim_background = config.trim_background
        self.metadata_path = config.metadata_path
        self.add_xyz = config.add_xyz

        self.metadata = np.load(self.metadata_path, allow_pickle=True).item()
        metadata_key = 'train' if self.train else 'test'
        self.labels = self.metadata[metadata_key]

        self.paths = [os.path.join(self.root, f'tensor_{k}') for k in self.labels]
        self.features = config.features 

    def __getitem__(self, idx):

        try:
            tensor_dict = torch.load(self.paths[idx])
        except:
            set_trace()
            
        label_tensor_torch = tensor_dict['label']
        mask_tensor_torch = None
        if 'mask' in tensor_dict.keys():
            mask_tensor_torch = tensor_dict['mask']

        brain_tensor_torch = torch.stack([tensor_dict[f] for f in self.features], dim=0)

        if self.trim_background:
            brain_tensor_torch, label_tensor_torch, mask_tensor_torch = trim(brain_tensor_torch, 
                                                                             label_tensor_torch,
                                                                             mask_tensor_torch)

        return brain_tensor_torch, \
                mask_tensor_torch.unsqueeze(0), \
                label_tensor_torch.unsqueeze(0)

    def __len__(self):
        return len(self.labels)
