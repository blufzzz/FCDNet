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
from tqdm import tqdm
from easydict import EasyDict as edict


def create_datasets(config):

    if config.dataset.dataset_type == 'interpolated': 
        train_dataset = BrainMaskDataset(config.dataset, train=True)
        val_dataset = BrainMaskDataset(config.dataset, train=False)
    elif config.dataset.dataset_type == 'brats_interpolated': 
        train_dataset = Brats2020Dataset(config.dataset, train=True)
        val_dataset = Brats2020Dataset(config.dataset, train=False)
    elif config.dataset.dataset_type == 'patches_tio':
        train_dataset = BrainMaskDataset(config.dataset, train=True)
        val_dataset = BrainMaskDataset(config.dataset, train=False)
    elif config.dataset.dataset_type == 'patches_prec':
        train_dataset = PatchesDataset(config.dataset, train=True)
        val_dataset = BrainMaskDataset(config.dataset, train=False)    
    else:
        raise RuntimeError('Wrond `dataset_type`!')
    return train_dataset, val_dataset


class PatchesDataset(Dataset):

    def __init__(self, config, force_rebuild=False, train=True):
        
        self.root = config.root
        self.train = train
        self.features = config.features
        self.patch_size = config.patch_size
        self.fcd_threshold = config.fcd_threshold
        self.metadata_path = config.metadata_path
        self.patch_dataframe_path = config.patch_dataframe_path
        self.patches_dataframes_merged_path = config.patches_dataframes_merged_path 
        self.metadata = np.load(self.metadata_path, allow_pickle=True).item()
        metadata_key = 'train' if self.train else 'test'
        self.labels = self.metadata[metadata_key]
        self.make_balanced_resampling = config.make_balanced_resampling
        self.force_rebuild = force_rebuild

        dataset_title = f'ps{self.patch_size}_fcd{self.fcd_threshold}'
        if self.make_balanced_resampling:
            dataset_title += '_balanced_resampling'

        df_path = os.path.join(self.patches_dataframes_merged_path, dataset_title)

        if self.force_rebuild or not os.path.isfile(df_path):
            df_s = []
            print('Creating patches dataframe for train dataset...')
            for k in tqdm(self.labels):
                df_label = f'label-{k}_ps{self.patch_size}_notrim'
                path = os.path.join(self.patch_dataframe_path, df_label)
                df_s.append(self.process_df(pd.read_csv(path, index_col=0)))
            self.df = pd.concat(df_s)
            del df_s
            print('Merged df created and saved to', df_path)
            self.df.to_csv(df_path)

        else:
            print('Reading pre-calculated dataframe from', df_path)
            self.df = pd.read_csv(df_path, 
                                  index_col=0,
                                  dtype={'x':int,'y':int,'z':int,
                                  'label':str,
                                  'target':int})

        print('Dataframe created, Class balance:', self.df['target'].sum()/self.df.shape[0])

    def balanced_resampling(self, df):
        # print('Rebalancing, orig shape:', df.shape)
        target_ind = np.array(df.query('target==1').index.tolist())
        non_target_ind = df.query('target!=1').index
        non_target_ind_sample = np.random.choice(non_target_ind, size=len(target_ind), replace=False)
        new_indexes = np.concatenate([target_ind,non_target_ind_sample])
        new_indexes = pd.core.indexes.numeric.Int64Index(data=new_indexes)
        df_resampled = df.loc[new_indexes]
        return df_resampled
        
    def process_df(self, df):
            
        df['fcd_percentage'] = df['n_label'] / df['n_fcd']
        df['target'] = (df['fcd_percentage'] >= self.fcd_threshold).astype(int)
        drop_index = df.query(f"fcd_percentage>0 & fcd_percentage<{self.fcd_threshold}").index
        df.drop(index=drop_index, inplace=True)
        if self.make_balanced_resampling:
            df = self.balanced_resampling(df)
        return df[['x','y','z', 'label', 'target']]

    def __getitem__(self, idx):

        # if self.make_balanced_resampling:
        #     info = self.df_resampled.iloc[idx]
        # else:
        info = self.df.iloc[idx]
        x = info.x
        y = info.y
        z = info.z
        label = info.label
        target = torch.tensor([info.target])
        
        tensor_path = os.path.join(self.root, f'tensor_{label}')
        tensor_dict = torch.load(tensor_path)
        label_tensor_torch = tensor_dict['label']
        
        mask_tensor_torch = None
        if 'mask' in tensor_dict.keys():
            mask_tensor_torch = tensor_dict['mask']
        
        if self.features == 'ALL':
            self.features = set(tensor_dict.keys()) - {'label', 'mask'}
        
        brain_tensor_torch = torch.stack([tensor_dict[f] for f in self.features], dim=0)
        
        pd = self.patch_size//2
        x1,x2 = x-pd,x+pd 
        y1,y2 = y-pd,y+pd
        z1,z2 = z-pd,z+pd
        
        patch = brain_tensor_torch[:,x1:x2,y1:y2,z1:z2]
        
        return patch, target

    def __len__(self):
        return self.df.shape[0]

        # if self.make_balanced_resampling:
        #     return self.df_resampled.shape[0]
        # else:
        #     return s


class Brats2020Dataset(Dataset):

    def __init__(self, config, train=True):
        self.root = config.root
        self.train = train
        self.trim_background = config.trim_background

        self.metadata = np.load('metadata_brats2020.npy',allow_pickle=True).item()
        metadata_key = 'train' if self.train else 'test'
        self.labels = self.metadata[metadata_key]

        self.paths = [os.path.join(self.root, f'BraTS20_Training_{k}/BraTS20_Training_{k}') for k in self.labels]
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
            
        return brain_tensor_torch,\
                mask_tensor_torch.unsqueeze(0),\
                label_tensor_torch.unsqueeze(0)

    def __len__(self):
        return len(self.paths)


class BrainMaskDataset(Dataset):


    def __init__(self, config, train=True):
        self.root = config.root
        self.train = train
        self.trim_background = config.trim_background
        self.metadata_path = config.metadata_path

        self.metadata = np.load(self.metadata_path, allow_pickle=True).item()
        metadata_key = 'train' if self.train else 'test'
        self.labels = self.metadata[metadata_key]#[:1]

        self.paths = [os.path.join(self.root, f'tensor_{k}') for k in self.labels]
        self.features = config.features 

    def __getitem__(self, idx):

        tensor_dict = torch.load(self.paths[idx])
        label_tensor_torch = tensor_dict['label']
        mask_tensor_torch = None
        if 'mask' in tensor_dict.keys():
            mask_tensor_torch = tensor_dict['mask']

        if self.features == 'ALL':
            self.features = set(tensor_dict.keys()) - {'label', 'mask'}

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
