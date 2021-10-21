import numpy as np
from matplotlib import pyplot as plt
import torch
from IPython.core.debugger import set_trace

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




