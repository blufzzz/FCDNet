import torch
from IPython.core.debugger import set_trace
from collections import defaultdict
import glob
from IPython.core.display import HTML
import os
import nibabel
import argparse
import re
import numpy as np
from matplotlib import pyplot as plt

def get_label(path):
    '''
    Extracts label from path, e.g.:
    '/workspace/RawData/Features/preprocessed_data/label_bernaskoni/n16.nii.gz' -> 'n16'
    '''
    return path.split('/')[-1].split('.')[0]


def show_prediction_slice(i, brain_tensor, mask_tensor, label_tensor, label_tensor_predicted, b_ind=0, c_ind=0):
    
    '''
    b_ind - batch_index
    c_ind - channel index for `brain_tensor`
    brain_tensor - [bs,C,1,H,W,D]
    mask_tensor - [bs,1,1,H,W,D]
    label_tensor - [bs,1,1,H,W,D]
    label_tensor_predicted - [bs,1,1,H,W,D]
    '''
    
    label_pos = (label_tensor[b_ind,0] > 0).sum(dim=(0,1)).argmax().item()
    
    fig = plt.figure("image", (3*5, 5), dpi=100)
    
    ax1 = plt.subplot(1, 3, 1)
    ax1.imshow(to_numpy(brain_tensor[b_ind,c_ind,:,:,label_pos]), cmap='gray')
    ax1.imshow(to_numpy(label_tensor[b_ind,0,:,:,label_pos]), alpha=0.6, cmap='Reds')
    
    ax2 = plt.subplot(1, 3, 2)
    ax2.imshow(to_numpy(brain_tensor[b_ind,c_ind,:,:,label_pos]), cmap='gray')
    ax2.imshow(to_numpy(label_tensor_predicted[b_ind,0,:,:,label_pos]), alpha=0.6, cmap='Reds')
    
    ax3 = plt.subplot(1, 3, 3)
    ax3.imshow(to_numpy(brain_tensor[b_ind,c_ind+1,:,:,label_pos]), cmap='jet')
    
    plt.xticks([])
    plt.yticks([])
    
    plt.savefig(f'inference_img/val_inference_{i}.png')


def get_latest_weights(logdir, number=None):
    
    checkpoints_path = os.path.join(logdir, 'checkpoints')
    
    if number is None:
        checkpoints_names = os.listdir(checkpoints_path)
        checkpoints_names = sorted(checkpoints_names, key=lambda x: int(re.findall('\d+', x)[0]))
        checkpoint = checkpoints_names[-1]
    else:
        checkpoint = f'weights_{number}.pth'
    
    return os.path.join(checkpoints_path, checkpoint)


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


def normalize_(brain_tensor, a_min_max=None):
    if a_min_max is None:
        a_min = brain_tensor.min()
        a_max = brain_tensor.max()
    else:
        a_min, a_max = a_min_max
    return (brain_tensor - a_min) / (a_max - a_min)

def normalize(brain_tensor, mask=None, a_min_max=None):
    
    ndim = len(brain_tensor.shape)

    if mask is None:
        background = brain_tensor[0,0,0]
        brain_tensor = brain_tensor - background # make background-level pixel to be zero
        mask = brain_tensor != 0
    else:
        brain_tensor[~mask] = 0
        
    brain_tensor[mask] = normalize_(brain_tensor[mask], a_min_max=a_min_max)
    
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
