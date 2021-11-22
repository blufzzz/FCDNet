import numpy as np
from matplotlib import pyplot as plt
import torch
from IPython.core.debugger import set_trace
from collections import defaultdict
import glob
import os
import nibabel

def create_dicts(root_label,
                 root_data,
                 root_geom_features, 
                 USE_GEOM_FEATURES, 
                 GEOM_FEATURES):
    
    paths_dict = defaultdict(dict)
    for p in os.listdir(root_label):

        label = p.split('.')[0]
        
        sub_root = os.path.join(root_data, f'sub-{label}/anat/')
            
        brain_path = glob.glob(os.path.join(sub_root, '*Asym_desc-preproc_T1w.nii.gz'))[0]
        mask_path = glob.glob(os.path.join(sub_root, '*Asym_desc-brain_mask.nii.gz'))[0]
        label_path = os.path.join(root_label, p) 

        paths_dict[label]['label'] = label_path
        paths_dict[label]['brain'] = brain_path    
        paths_dict[label]['mask'] = mask_path  

        # features

        if USE_GEOM_FEATURES:
            for g in GEOM_FEATURES:
                g_path = os.path.join(root_geom_features, f'{g}/norm-{label}.nii')
                paths_dict[label][g] = g_path    
                
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
                patch_size):
    
    X,Y,Z = mask_tensor.shape

    x1,x2 = x-patch_size//2, x+patch_size//2
    y1,y2 = y-patch_size//2, y+patch_size//2
    z1,z2 = z-patch_size//2, z+patch_size//2

    if (np.array([x1,x2,y1,y2,z1,z2]) < 0).any():
        return None

    else:

        volume_mask= mask_tensor[x1:x2,y1:y2,z1:z2]
        volume_label = label_tensor[x1:x2,y1:y2,z1:z2]

        p_mask = volume_mask.sum()/np.prod(volume_mask.shape)
        p_label = volume_label.sum()/np.prod(volume_label.shape)

        return [x,y,z,p_mask,p_label]
    
def get_symmetric_value(a, a_sym):
    diff = a-a_sym
    return a_sym - diff


def pad_arrays(arrays_list, padding_size):
    return [np.pad(array,((padding_size,padding_size),
                                (padding_size,padding_size),
                                (padding_size,padding_size))) for array in arrays_list]




