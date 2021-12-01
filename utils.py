import numpy as np
from matplotlib import pyplot as plt
import torch
from IPython.core.debugger import set_trace
from collections import defaultdict
import glob
from celluloid import Camera
from IPython.core.display import HTML
import os
import nibabel

def video_comparison(brain_tensor, mask_tensor_real, mask_tensor_pred, n_slides=64):
    
    fig, ax = plt.subplots(1,2)
    X_max, Y_max, Z_max = brain_tensor.shape
    camera = Camera(fig)
    
    for i in range(n_slides):
        
        y_slice_pos = (Y_max//(n_slides+2))*(i+1)
        
        brain_tensor_slice = brain_tensor[:,y_slice_pos,:]
        ax[0].imshow(brain_tensor_slice, 'gray')
        ax[1].imshow(brain_tensor_slice, 'gray')
        
        mask_tensor_real_slice = mask_tensor_real[:,y_slice_pos,:]
        mask_tensor_pred_slice = mask_tensor_pred[:,y_slice_pos,:]
        
        ax[0].imshow(mask_tensor_real_slice, 'jet', interpolation='none', alpha=0.7)
        ax[1].imshow(mask_tensor_pred_slice, 'jet', interpolation='none', alpha=0.7)
        
        ax[0].set_title('GT')
        ax[1].set_title('Pred')
        
        camera.snap()
        
    return camera   



def trim(brain_tensor, mask_tensor, label_tensor):
    '''
    mask_tensor - [H,W,D]
    brain_tensor - [N_features, H,W,D]
    label_tensor - [H,W,D]
    
    '''
    X,Y,Z = mask_tensor.shape
    
    X_mask = mask_tensor.sum(dim=[1,2]) > 0
    Y_mask = mask_tensor.sum(dim=[0,2]) > 0
    Z_mask = mask_tensor.sum(dim=[0,1]) > 0
    
    brain_tensor_trim = brain_tensor[:,X_mask][:,:,Y_mask][:,:,:,Z_mask]
    mask_tensor_trim = mask_tensor[X_mask][:,Y_mask][:,:,Z_mask]    
    label_tensor_trim = label_tensor[X_mask][:,Y_mask][:,:,Z_mask]
    
    return brain_tensor_trim, mask_tensor_trim, label_tensor_trim

def create_dicts(root_label,
                 root_data,
                 root_geom_features,
                 allowed_keys=None, 
                 USE_GEOM_FEATURES=False, 
                 GEOM_FEATURES=None):
    
    keys = [p.split('.')[0] for p in os.listdir(root_label)]
    if allowed_keys is not None:
        keys = set(keys).intersection(set(allowed_keys))

    paths_dict = defaultdict(dict)
    for label in keys:

        sub_root = os.path.join(root_data, f'sub-{label}/anat/')
            
        brain_path = glob.glob(os.path.join(sub_root, '*Asym_desc-preproc_T1w.nii.gz'))[0]
        mask_path = glob.glob(os.path.join(sub_root, '*Asym_desc-brain_mask.nii.gz'))[0]
        label_path = os.path.join(root_label, label + '.nii')   

        # features

        if USE_GEOM_FEATURES:
            try:
                for g in GEOM_FEATURES:
                    g_path = os.path.join(root_geom_features, f'{g}/norm-{label}.nii')
                    if not os.path.isfile(g_path):
                        raise RuntimeError
                    else:
                        paths_dict[label][g] = g_path 
            except RuntimeError:
                continue

        paths_dict[label]['label'] = label_path
        paths_dict[label]['brain'] = brain_path    
        paths_dict[label]['mask'] = mask_path   
                
    return paths_dict



def normalized(brain_tensor):
    
    return (brain_tensor - brain_tensor.min()) / (brain_tensor.max() - brain_tensor.min())

def load(path_dict):

    mask_tensor = nibabel.load(path_dict['mask']).get_fdata() > 0
    mask_tensor_int = mask_tensor.astype(int) 
    brain_tensor = nibabel.load(path_dict['brain']).get_fdata() * mask_tensor_int
    label_tensor = nibabel.load(path_dict['label']).get_fdata() * mask_tensor_int
    label_tensor = (label_tensor > 0).astype('int')

    # SHAPE = label_tensor.shape

    # geom_features = []
    # for k,v in path_dict.items():
    #     if k not in ['mask', 'brain', 'label']:
    #         geom_features.append()
    #     assert SHAPE == v.shape

    
    return [brain_tensor, mask_tensor, label_tensor]


def show_slices(brain_tensor, n_slices_show=5, mask_tensor=None):
    
    fig, axes = plt.subplots(ncols=3, nrows=n_slices_show, figsize=(15,n_slices_show*5))
    X_max, Y_max, Z_max = brain_tensor.shape
    for i in range(n_slices_show):

        x_slice_pos = (X_max//(n_slices_show+2))*(i+1)
        y_slice_pos = (Y_max//(n_slices_show+2))*(i+1)
        z_slice_pos = (Z_max//(n_slices_show+2))*(i+1)

        brain_tensor_x_slice = brain_tensor[x_slice_pos,:,:]
        brain_tensor_y_slice = brain_tensor[:,y_slice_pos,:]
        brain_tensor_z_slice = brain_tensor[:,:,z_slice_pos]

        axes[i,0].imshow(brain_tensor_x_slice, 'gray')
        axes[i,1].imshow(brain_tensor_y_slice, 'gray')
        axes[i,2].imshow(brain_tensor_z_slice, 'gray')
        
        if mask_tensor is not None:
            
            mask_tensor_x_slice = mask_tensor[x_slice_pos,:,:]
            mask_tensor_y_slice = mask_tensor[:,y_slice_pos,:]
            mask_tensor_z_slice = mask_tensor[:,:,z_slice_pos]

            axes[i,0].imshow(mask_tensor_x_slice, 'jet', interpolation='none', alpha=0.7)
            axes[i,1].imshow(mask_tensor_y_slice, 'jet', interpolation='none', alpha=0.7)
            axes[i,2].imshow(mask_tensor_z_slice, 'jet', interpolation='none', alpha=0.7)

    plt.tight_layout()
    plt.show()


# TODO: add check that FCD is not on boundary!
def check_patch(x,y,z,
                mask_tensor, 
                label_tensor, 
                pad):
    
    x1,x2 = x-pad, x+pad
    y1,y2 = y-pad, y+pad
    z1,z2 = z-pad, z+pad

    if (np.array([x1,x2,y1,y2,z1,z2]) < 0).any():
        return None

    else:
        volume_mask= mask_tensor[x1:x2,y1:y2,z1:z2]
        volume_label = label_tensor[x1:x2,y1:y2,z1:z2]

        p_mask = volume_mask.sum()/np.prod(volume_mask.shape)
        n_label = volume_label.sum()

        patch_info = {'x':x,
                      'y':y,
                      'z':z,
                      'p_mask':p_mask,
                      'n_label':n_label}

        return patch_info
    
def get_symmetric_coordinate(r):
    diff = a-a_sym
    return a_sym - diff


def pad_arrays(arrays_list, padding_size):
    return [np.pad(array,((padding_size,padding_size),
                                (padding_size,padding_size),
                                (padding_size,padding_size))) for array in arrays_list]




