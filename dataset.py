import os, re
from collections import defaultdict
import numpy as np
from monai.data import create_test_image_3d, list_data_collate, decollate_batch, pad_list_data_collate
import monai
from monai.data import DataLoader, Dataset 
from monai.transforms import (
    LoadImage, Spacingd, RandZoomd,
    RandFlipd, Resized, RandAffined,
    LoadImaged, EnsureChannelFirstd,
    Resized, EnsureTyped, Compose, ScaleIntensityd, 
    RandGaussianNoised, RandRotated
)

from monai.transforms.intensity.array import ScaleIntensity
import torch
from IPython.core.debugger import set_trace
from torchio.transforms.preprocessing.intensity import histogram_standardization
from torchio.transforms.preprocessing.intensity import z_normalization
from pathlib import Path

from utils import normalize, normalize_

BASE_DIR = '/workspace/RawData/Features'

FEATURES_LIST = ['image', 't2', 'flair', 'blurring-t1', 'blurring-t2', 'blurring-Flair', 'cr-t2', 'cr-Flair', 'thickness', 'curv', 'sulc', 'variance', 'entropy']


def assign_feature_maps(sub, feature):
    '''
    Mapping from `sub` and `feature` to the corresponding path
    of feature that belongs ot this subject 
    the list of possible paths may be returned instead 
    of single path
    '''
    global BASE_DIR
    if feature == 'image':
        feature_map = os.path.join(BASE_DIR, f'prep_wf', f'sub-{sub}', f'sub-{sub}_t1_brain-final.nii.gz')
        
    elif feature == 't2':
        feature_map = os.path.join(BASE_DIR, f'prep_wf', f'sub-{sub}', f'sub-{sub}_t2_brain-final.nii.gz')
        
    elif feature == 'flair':
        feature_map = os.path.join(BASE_DIR, f'prep_wf', f'sub-{sub}', f'sub-{sub}_fl_brain-final.nii.gz')
        
    elif feature == 'blurring-t1':
        feature_map = os.path.join(BASE_DIR, f'prep_wf', f'sub-{sub}', f'Blurring_T1.nii.gz')
        
    elif feature == 'blurring-t2':
        feature_map = os.path.join(BASE_DIR, f'prep_wf', f'sub-{sub}', f'Blurring_T2.nii.gz')
        
    elif feature == 'blurring-Flair':
        feature_map = os.path.join(BASE_DIR, f'prep_wf', f'sub-{sub}', f'Blurring_Flair.nii.gz')
        
    elif feature == 'cr-t2':
        feature_map = os.path.join(BASE_DIR, f'prep_wf', f'sub-{sub}', f'CR_T2.nii.gz')
        
    elif feature == 'cr-Flair':
        feature_map = os.path.join(BASE_DIR, f'prep_wf', f'sub-{sub}', f'CR_Flair.nii.gz')
                       
    elif feature == 'thickness':
        feature_map = os.path.join(BASE_DIR, f'prep_wf', f'sub-{sub}', f'thickness_mni.nii')
        
    elif feature == 'curv':
        feature_map = os.path.join(BASE_DIR, f'prep_wf', f'sub-{sub}', f'curv_mni.nii')
        
    elif feature == 'sulc':
        feature_map = os.path.join(BASE_DIR, f'prep_wf', f'sub-{sub}', f'sulc_mni.nii')
        
    elif feature == 'variance':
        feature_map = os.path.join(BASE_DIR, f'prep_wf', f'sub-{sub}', f'Variance.nii.gz')
        
    elif feature == 'entropy':
        feature_map = os.path.join(BASE_DIR, f'prep_wf', f'sub-{sub}', f'Entropy.nii.gz')
        
    elif feature == 'mask':
        feature_map = os.path.join(BASE_DIR, f'prep_wf', f'sub-{sub}', f'sub-{sub}_t1_brain-final_mask.nii.gz')
    
    return feature_map


def create_datafile(sub_list, feat_params, mask=False):
    
    '''
    for each subject from `sub_list` 
    collects corresponding features from `feat_params`
    and segmentation mask
    
    mask : Bool,
        Include mask key to output file
    '''

    files = []
    missing_files = defaultdict(list)

    for sub in sub_list:
        
        images_per_sub = dict()
        images_per_sub['image'] = []
        if mask:
            images_per_sub['mask'] = []
            mask_path = assign_feature_maps(sub, 'mask')    
            if os.path.isfile(mask_path):
                images_per_sub['mask'] = mask_path
            else:
                missing_files['mask'].append(mask_path)
        
        for feat in feat_params:
            proposed_map_paths = assign_feature_maps(sub, feat)
            map_path = None # path of the `feat`
                
            # in case `proposed_map_paths` is single path
            if not isinstance(proposed_map_paths, list):
                proposed_map_paths = [proposed_map_paths]
            
            for proposed_map_path in proposed_map_paths:
                if os.path.isfile(proposed_map_path):
                    map_path = proposed_map_path
            
            if map_path is not None:
                # feature path found and added to `image` field 
                images_per_sub['image'].append(map_path)
            else:
                missing_files['image'].append(proposed_map_path)
        
        seg_path = os.path.join(BASE_DIR, 'preprocessed_data/label_bernaskoni', f'{sub}.nii.gz')
        
        if os.path.isfile(seg_path):
            images_per_sub['seg'] = seg_path
        else:
            missing_files['seg'].append(seg_path)
            
        files.append(images_per_sub)
        
    return files, missing_files



def setup_datafiles(split_dict, config):

    '''
    split_dict - dict:{'train':[...], 
                       'test':[...]}, train-test split
    returns: train_files, val_files: lists of dicts, corresponding to subjects
    each dict in <>_files list looks like: 
        {'image':[path_feature1, path_feature2,...], 'seg':segpath, 'mask':maskpath}
    '''
    
    train_list = split_dict.get('train')
    val_list = split_dict.get('test')
    
    images_list = []
    feat_params = config.dataset.features
    
    # Flag to add mask as additional sequence to Subset
    add_mask = config.dataset.trim_background
    
    train_files, train_missing_files = create_datafile(train_list, feat_params, mask=add_mask)
    val_files, val_missing_files = create_datafile(val_list, feat_params, mask=add_mask)
    
    print(f"Train set length: {len(train_files)}\nTest set length: {len(val_files)}")
    
    # print missing files
    total_missing_files = 0
    for split_name, missing_files_dict  in {'train': train_missing_files, 
                                            'val': val_missing_files}.items():
        
        for feature_type, missing_features_list in missing_files_dict.items():
    
            print(f'{split_name} missing {feature_type}:')
            for fpath in missing_features_list:
                print(fpath)
            total_missing_files += len(missing_features_list)
    
    assert total_missing_files == 0
    
    return train_files, val_files


def scaling_specified(data_dict, features, scaling_dict):
    '''
    features - list of features e.g. ['image', 'curv', 'sulc',...]
    scaling_dict - {
                    'feature_name_1': [a, b], - use provided `a` and `b` for (x-a)/b normalization
                    'feature_name_2': None, - infer `a_min` and `a_max` from the data for min-max normalization
                   }
    '''
    mask_bool = data_dict["mask"][0] > 0.
    for i, feature in enumerate(features):
        v = scaling_dict[feature]
        data_dict["image"][i][mask_bool] = normalize_(data_dict["image"][i][mask_bool], ab=v)
    return data_dict

def scaling_specified_wrapper(features, scaling_dict):
    '''
    decorate `minmax_scaling_specified` with pre-specified `scaling_dict`
    '''
    def wrapper(data_dict):
        return scaling_specified(data_dict, features=features, scaling_dict=scaling_dict)
    return wrapper

def scaling_as_torchio(data_dict, features, scaling_dict):
    mask_bool = data_dict["mask"] > 0.
    features_ = features
    for i, feature in enumerate(features_):
        #  condition, beceause some features like curv, sulc, thickness - don't need in scale, however, can be done.
        if feature not in ['blurring-t1', 'blurring-t2', 'blurring-Flair', 'cr-t2', 'cr-Flair', 'variance', 'entropy']:
            landmarks_path = Path(f'/workspace/FCDNet/landmarks/{feature}_landmarks.npy')
        else:
            landmarks_path = Path(f'/workspace/FCDNet/landmarks/{feature}_False_landmarks.npy')
        landmark =  np.load(landmarks_path)
        d = torch.tensor(data_dict["image"][i])
        m = torch.tensor(mask_bool)
        d_n = histogram_standardization._normalize(d, landmark, m)
        tensor = z_normalization.ZNormalization.znorm(d_n, m)
        if tensor is not None:
            data_dict["image"][i] = tensor
    return data_dict

def scaling_torchio_wrapper(features, scaling_dict):
    '''
    decorate `minmax_scaling_specified` with pre-specified `scaling_dict`
    '''
    def wrapper(data_dict):
        return scaling_as_torchio(data_dict, features=features, scaling_dict=scaling_dict)
    return wrapper
    
def binarize_target(data_dict, eps=1e-3):
    '''
    data_dict - [C,H,W,D]
    '''
    data_dict["seg"] = (data_dict["seg"] > eps).astype(data_dict["seg"].dtype)
    return data_dict
    
    
def mask_transform(data_dict):
    '''
    data_dict - [C,H,W,D]
    '''
    data_dict["mask"] = (data_dict["mask"] > 0).astype(data_dict["image"].dtype)
    data_dict["image"] = data_dict["image"] * (data_dict["mask"])
    return data_dict

def setup_transformations(config, scaling_dict=None):
    
    interpolate = config.default.interpolate
    if interpolate:
        spatial_size_conf = tuple(config.default.interpolation_size)
    features = config.dataset.features
    
    assert config.dataset.trim_background
    keys=["image", "seg", "mask"]
    sep_k=["seg", "mask"]
    
    if scaling_dict in 'torchio':
        scaler = scaling_torchio_wrapper(features, scaling_dict)
    elif scaling_dict in 'scale_metadata':
        scaler = scaling_specified_wrapper(features, scaling_dict)
    else:
        scaler = ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0, channel_wise=True)
        
    # no-augmentation transformation
    val_transf = Compose(
                        [
                            LoadImaged(keys=keys),
                            EnsureChannelFirstd(keys=keys)
                        ] + ([Resized(keys=keys, spatial_size=spatial_size_conf)] if interpolate else []) + \
                        [
                            Spacingd(keys=sep_k, pixdim=1.0),
                            mask_transform, # zero the non-mask values
                            binarize_target,
                            scaler,
                            EnsureTyped(keys=sep_k, dtype=torch.float),
                        ]
                        )
        
    if config.opt.augmentation:
        
        rand_affine_prob = config.opt.rand_affine_prob
        rot_range = config.opt.rotation_range
        shear_range = config.opt.shear_range
        scale_range = config.opt.scale_range
        translate_range = config.opt.translate_range

        noise_std = config.opt.noise_std
        flip_prob = config.opt.flip_prob
        rand_zoom_prob = config.opt.rand_zoom_prob
        
        # basic operations
        transforms = [LoadImaged(keys=keys), 
                      EnsureChannelFirstd(keys=keys),
                      
                     ] + ([Resized(keys=keys, spatial_size=spatial_size_conf)] if interpolate else []) + \
                     [mask_transform,scaler, Spacingd(keys=sep_k, pixdim=1.0)]
        
        if rand_affine_prob == 0 and rot_range > 0:
            transforms.append(RandRotated(keys=keys, # apply to all!
                                range_x=rot_range, 
                                range_y=rot_range, 
                                range_z=rot_range, 
                                prob=0.5)
                             )
        if flip_prob > 0:
            transforms.append(RandFlipd(keys=keys, # apply to all!
                                        prob=flip_prob, 
                                        spatial_axis=0))
            
        if rand_affine_prob > 0:
            transforms.append(RandAffined(prob=rand_affine_prob, 
                                         rotate_range=[rot_range, rot_range, rot_range], 
                                         shear_range=[shear_range, shear_range, shear_range], 
                                         translate_range=[translate_range, translate_range, translate_range], 
                                         scale_range=[scale_range, scale_range, scale_range], 
                                         padding_mode='zeros',
                                         keys=keys # apply to all!
                                        )
                             )

        if noise_std > 0:
            transforms.append(RandGaussianNoised(prob=0.5, 
                                                mean=0.0, 
                                                std=noise_std, 
                                                keys=["image"]
                                               )
                             )
                              
        if rand_zoom_prob > 0:
            transforms.append(RandZoomd(prob=0.5, min_zoom=0.9, max_zoom=1.1, keys=keys))
        
        # add the rest 
        transforms.extend([ # zero the non-mask values
                            binarize_target,
                            EnsureTyped(keys=sep_k, dtype=torch.float),
                         ]
                        )
        
        train_transf = Compose(transforms)
    else:
        train_transf = val_transf
    
    return train_transf, val_transf


def setup_dataloaders(config, pin_memory=True):
    
    # load metadata: train-test split dict
    metadata_path = config.dataset.metadata_path
    
    scaling_dict = None
    if config.dataset.scaling_method in 'torchio':
        scaling_dict = 'torchio'
    elif config.dataset.scaling_method in 'scale_metadata':
        
        scaling_data_path = config.dataset.scaling_metadata_path
        scaling_dict = np.load(scaling_data_path, allow_pickle=True).item()
    else:
        print('Warning! no SCALING METADATA used! Applying naive independent MinMax...')
    
    split_dict = np.load(metadata_path, allow_pickle=True).item()   
    
    train_files, val_files = setup_datafiles(split_dict, config)
    print(f"Scaling Dict: {scaling_dict}")
    train_transf, val_transf = setup_transformations(config, scaling_dict)
    
    # training dataset
    train_ds = monai.data.Dataset(data=train_files, transform=train_transf)
    train_loader = DataLoader(
        train_ds,
        batch_size=config.opt.train_batch_size,
        shuffle=config.dataset.shuffle_train,
        num_workers=0,
        collate_fn=list_data_collate,
        pin_memory=torch.cuda.is_available() and pin_memory
    )

    # validation dataset
    val_ds = monai.data.Dataset(data=val_files, transform=val_transf)
    val_loader = DataLoader(val_ds, 
                            batch_size=config.opt.val_batch_size, 
                            shuffle=False, # important not to shuffle, to ensure label correspondence
                            num_workers=0, 
                            collate_fn=list_data_collate,
                            pin_memory=torch.cuda.is_available() and pin_memory
                            )
    
    return train_loader, val_loader