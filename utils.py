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




def create_dicts(root_label,
                 feature_paths_templates,
                 label_extractor, 
                 broken_labels=None):

    '''
    feature_paths_templates - dict; feature_type:template
    label_extractor - extracts label unique for each subj
    t1:/sub-{label}/anat/
    '''
    
    keys = [label_extractor(p) for p in os.listdir(root_label)]
    if broken_labels is not None:
        keys = set(keys)-set(broken_labels)

    paths_dict = defaultdict(dict)
    for label in keys:
        for feature_type, template in feature_paths_templates.items():
            
            path = template.replace('{label}', label)
            
            # no such path
            try:
                path = glob.glob(path, recursive=True)[0]
            except:
                print(f'No {feature_type} for {label}')
                if label in paths_dict.keys():
                    paths_dict.pop(label)
                break

            paths_dict[label][feature_type] = path

    return paths_dict


def normalize_(brain_tensor):
    return (brain_tensor - brain_tensor.min()) / (brain_tensor.max() - brain_tensor.min())

def normalize(brain_tensor, mask=None):

    ndim = len(brain_tensor.shape)

    # if ndim == 4
    
    if mask is None:
        background = brain_tensor[0,0,0]
        brain_tensor = brain_tensor - background # make background-level pixel to be zero
        mask = brain_tensor != background

    brain_tensor[~mask] = 0
    brain_tensor[mask] = normalize_(brain_tensor[mask])
    
    return brain_tensor


def load(path_dict):
    results = {}
    for k,v in path_dict.items():
        results[k] = nibabel.load(v).get_fdata()
    return results



# TODO: add check that FCD is not on boundary!
def check_patch(x,y,z,
                mask_tensor, 
                label_tensor, 
                pad,
                p_thresh=None):
    
    x1,x2 = x-pad, x+pad
    y1,y2 = y-pad, y+pad
    z1,z2 = z-pad, z+pad

    if (np.array([x1,x2,y1,y2,z1,z2]) < 0).any():
        return None

    else:
        volume_mask= mask_tensor[x1:x2,y1:y2,z1:z2]
        p_mask = volume_mask.sum()/np.prod(volume_mask.shape)

        if p_thresh is not None and p_mask < p_thresh:
            return None

        volume_label = label_tensor[x1:x2,y1:y2,z1:z2]
        n_label = volume_label.sum()

        patch_info = {'x':x,
                      'y':y,
                      'z':z,
                      'p_mask':p_mask,
                      'n_label':n_label}

        return patch_info

    
    
def pad_arrays(arrays_list, padding_size):
    return [np.pad(array,((padding_size,padding_size),
                                (padding_size,padding_size),
                                (padding_size,padding_size))) for array in arrays_list]




def save(experiment_dir, model, opt, epoch):
    checkpoint_dir = os.path.join(experiment_dir, "checkpoints") # , "{:04}".format(epoch)
    os.makedirs(checkpoint_dir, exist_ok=True)
    dict_to_save = {'model_state': model.state_dict(),'opt_state' : opt.state_dict(), 'epoch':epoch}
    torch.save(dict_to_save, os.path.join(checkpoint_dir, f"weights_{epoch}.pth"))


def DiceScoreBinary(input, 
                    target, 
                    include_backgroud=False, 
                    weights=None):
    '''
    Binary Dice score
    input - [bs,1,ps,ps,ps], probability [0,1]
    target - binary mask [bs,1,ps,ps,ps], 1 for foreground, 0 for background
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



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--logdir", type=str, default="")
    parser.add_argument('--experiment_comment', default='', type=str)
    args = parser.parse_args()
    return args