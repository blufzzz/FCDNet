import torch
from IPython.core.debugger import set_trace
from collections import defaultdict
import glob
from IPython.core.display import HTML
import os
import nibabel
import argparse
import re
from celluloid import Camera
import numpy as np
from matplotlib import pyplot as plt

def calc_gradient_norm(named_parameters):
    total_norm = 0.0
    for name, p in named_parameters:
        # print(name)
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2

    total_norm = total_norm ** (1. / 2)

    return total_norm


def get_capacity(model):
    s_total = 0
    for param in model.parameters():
        s_total+=param.numel()
    return round(s_total / (10**6),2)


def video(brain_tensor, mask_tensor=None, n_slides=100):

    '''
    brain_tensor - single ndarray brain [H,W,D]
    mask_tensor - single nndarray [H,W,D] masks to show ober the brain 
    '''
    
    fig, ax = plt.subplots()
    X_max, Y_max, Z_max = brain_tensor.shape
    camera = Camera(fig)
    
    for y_slice_pos in np.linspace(0,Y_max-1, n_slides, dtype=int):
        
        brain_tensor_slice = brain_tensor[:,y_slice_pos,:]
        ax.imshow(brain_tensor_slice, 'gray')
        
        if mask_tensor is not None:
            mask_tensor_slice = mask_tensor[:,y_slice_pos,:]
            ax.imshow(mask_tensor_slice, 'jet', interpolation='none', alpha=0.7)
        
        camera.snap()
        
    
    return camera   
         

def video_comparison(brains, masks=None, titles=None, n_slides=64):
    
    '''
    brains - list of ndarray [H,W,D] brains 
    masks - list of ndarray [H,W,D] masks to show ober the brain 

    '''
    
    N = len(brains)
    fig, ax = plt.subplots(1,N, sharex=True, sharey=True)
    X_max, Y_max, Z_max = brains[0].shape
    camera = Camera(fig)
    
    ax_iterator = ax if N > 1 else [ax]
    
    for y_slice_pos in np.linspace(0,Y_max-1, n_slides, dtype=int):
        
        for i,ax in enumerate(ax_iterator):
            
            brain_slice = brains[i][:,y_slice_pos,:]
            ax.imshow(brain_slice, 'gray')
            
            try:
                mask_slice = masks[i][:,y_slice_pos,:]
                ax.imshow(mask_slice, 'jet', interpolation='none', alpha=0.7)
            except:
                pass
            
            try:
                ax.set_title(titles[i])
            except:
                pass

        camera.snap()
        
    return camera     



def trim(brain_tensor, label_tensor, mask_tensor=None):

    '''
    mask_tensor - [H,W,D]
    brain_tensor - [N_features, H,W,D]
    label_tensor - [H,W,D]
    
    '''
    X,Y,Z = label_tensor.shape
    
    if mask_tensor is not None:
        X_mask = mask_tensor.sum(dim=[1,2]) > 0
        Y_mask = mask_tensor.sum(dim=[0,2]) > 0
        Z_mask = mask_tensor.sum(dim=[0,1]) > 0

        brain_tensor_trim = brain_tensor[:,X_mask][:,:,Y_mask][:,:,:,Z_mask]
        label_tensor_trim = label_tensor[X_mask][:,Y_mask][:,:,Z_mask]
        mask_tensor_trim = mask_tensor[X_mask][:,Y_mask][:,:,Z_mask]   

    else:
        background = 0
        assert (brain_tensor[:,0,0,0] == background).all()
        mask_tensor = (brain_tensor != background).sum(0)
        X_mask = mask_tensor.sum(dim=[1,2]) > 0
        Y_mask = mask_tensor.sum(dim=[0,2]) > 0
        Z_mask = mask_tensor.sum(dim=[0,1]) > 0

        x1,x2 = np.arange(X)[mask_tensor.sum(dim=[1,2]) > 0][[0,-1]]
        y1,y2 = np.arange(Y)[mask_tensor.sum(dim=[0,2]) > 0][[0,-1]]
        z1,z2 = np.arange(Z)[mask_tensor.sum(dim=[0,1]) > 0][[0,-1]]
    
        brain_tensor_trim = brain_tensor[:,x1:x2][:,:,y1:y2][:,:,:,z1:z2]
        label_tensor_trim = label_tensor[x1:x2][:,y1:y2][:,:,z1:z2]
        mask_tensor_trim = mask_tensor[x1:x2][:,y1:y2][:,:,z1:z2]   

    return brain_tensor_trim, label_tensor_trim, mask_tensor_trim


def normalize_(brain_tensor):
    return (brain_tensor - brain_tensor.min()) / (brain_tensor.max() - brain_tensor.min())

def normalize(brain_tensor, mask=None):

    ndim = len(brain_tensor.shape)

    if mask is None:
        background = brain_tensor[0,0,0]
        brain_tensor = brain_tensor - background # make background-level pixel to be zero
        mask = brain_tensor != background

    brain_tensor[~mask] = 0
    brain_tensor[mask] = normalize_(brain_tensor[mask])
    
    return brain_tensor

def get_label(path):
    '''
    Extracts label from path, e.g.:
    '/workspace/RawData/Features/preprocessed_data/label_bernaskoni/n16.nii.gz' -> 'n16'
    '''
    return path.split('/')[-1].split('.')[0]


def to_numpy(X):
    return X.detach().cpu().numpy()

def save(experiment_dir, model, opt, epoch):
    checkpoint_dir = os.path.join(experiment_dir, "checkpoints") # , "{:04}".format(epoch)
    os.makedirs(checkpoint_dir, exist_ok=True)
    dict_to_save = {'model_state': model.state_dict(),'opt_state' : opt.state_dict(), 'epoch':epoch}
    torch.save(dict_to_save, os.path.join(checkpoint_dir, f"weights_{epoch}.pth"))
