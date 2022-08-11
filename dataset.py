import os, re
from collections import defaultdict
import numpy as np
from monai.data import create_test_image_3d, list_data_collate, decollate_batch, pad_list_data_collate
import monai
from monai.data import DataLoader, Dataset 
from monai.transforms import (
    LoadImage, EnsureChannelFirst, Spacing,
    RandFlip, Resize, EnsureType,
    LoadImaged, EnsureChannelFirstd,
    Resized, EnsureTyped, Compose, ScaleIntensityd, 
    AddChanneld, MapTransform, AsChannelFirstd, EnsureType, 
    Activations, AsDiscrete, RandCropByPosNegLabeld, 
    RandRotate90d, LabelToMaskd, RandFlipd, RandRotated, Spacingd, RandAffined,
    RandShiftIntensityd, MaskIntensityd
)
from monai.transforms.intensity.array import ScaleIntensity
import torch
from IPython.core.debugger import set_trace

BASE_DIR = '/workspace/RawData/Features'

FEATURES_LIST = ['image', 't2', 'flair', 'blurring-t1', 'blurring-Flair', 'cr-t2', 'cr-Flair', 'thickness', 'curv', 'sulc', 'variance']


def masked_transform(data_dict):
    # set_trace()
    data_dict["image"] = data_dict["image"] * data_dict["mask"]
    return data_dict


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
        feature_map = os.path.join(BASE_DIR, f'prep_wf', f'sub-{sub}', f'Blurring_T2.nii.gz') # just addded
        
    elif feature == 'blurring-Flair':
        feature_map = os.path.join(BASE_DIR, f'prep_wf', f'sub-{sub}', f'Blurring_Flair.nii.gz')
        
    elif feature == 'cr-t2':
        feature_map = [os.path.join(BASE_DIR, f'prep_wf', f'sub-{sub}', f'CR_T2.nii'),
                       os.path.join(BASE_DIR, f'prep_wf', f'sub-{sub}', f'CR_T2.nii.gz')]
        
    elif feature == 'cr-Flair':
        feature_map = [os.path.join(BASE_DIR, f'prep_wf', f'sub-{sub}', f'CR_Flair.nii'),
                       os.path.join(BASE_DIR, f'prep_wf', f'sub-{sub}', f'CR_Flair.nii.gz')]
                       
    elif feature == 'thickness':
        feature_map = os.path.join(BASE_DIR, f'prep_wf', f'sub-{sub}', f'thickness_mni.nii')
        
    elif feature == 'curv':
        feature_map = os.path.join(BASE_DIR, f'prep_wf', f'sub-{sub}', f'curv_mni.nii')
        
    elif feature == 'sulc':
        feature_map = os.path.join(BASE_DIR, f'prep_wf', f'sub-{sub}', f'sulc_mni.nii')
        
    elif feature == 'variance':
        feature_map = os.path.join(BASE_DIR, f'prep_wf', f'sub-{sub}', f'Variance.nii.gz')
        
    elif feature == 'mask':
        feature_map = os.path.join(BASE_DIR, f'prep_wf', f'sub-{sub}', f'sub-{sub}_t1_brain-final_mask.nii.gz')
        
        #  Normalized feature maps
        
    elif feature == 'norm-blurring-t1':
        feature_map = os.path.join(BASE_DIR, f'prep_wf', f'sub-{sub}', f'norm-Blurring_T1.nii.gz')
        
    elif feature == 'norm-blurring-flair':
        feature_map = os.path.join(BASE_DIR, f'prep_wf', f'sub-{sub}', f'norm-Blurring_Flair.nii.gz')
    
    elif feature == 'norm-cr-t2':
        feature_map = os.path.join(BASE_DIR, f'prep_wf', f'sub-{sub}', f'norm-CR_T2.nii.gz')
        
    elif feature == 'norm-cr-flair':
        feature_map = os.path.join(BASE_DIR, f'prep_wf', f'sub-{sub}', f'norm-CR_Flair.nii.gz')
        
    elif feature == 'norm-variance':
        feature_map = os.path.join(BASE_DIR, f'prep_wf', f'sub-{sub}', f'norm-Variance.nii.gz')
        
    #  Postprocessed feature maps
        
    elif feature == 'post-blurring-t1':
        feature_map = os.path.join(BASE_DIR, f'postprocessing', f'sub-{sub}', f'post-Blurring_T1.nii.gz')
        
    elif feature == 'post-blurring-flair':
        feature_map = os.path.join(BASE_DIR, f'postprocessing', f'sub-{sub}', f'post-Blurring_Flair.nii.gz')
    
    elif feature == 'post-cr-t2':
        feature_map = os.path.join(BASE_DIR, f'postprocessing', f'sub-{sub}', f'post-CR_T2.nii.gz')
        
    elif feature == 'post-cr-flair':
        feature_map = os.path.join(BASE_DIR, f'postprocessing', f'sub-{sub}', f'post-CR_Flair.nii.gz')
        
    elif feature == 'post-variance':
        feature_map = os.path.join(BASE_DIR, f'postprocessing', f'sub-{sub}', f'post-Variance.nii.gz')
    
    else:
        raise KeyError(f'Wrong feature_map name: {feature}!')
        
        
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


def setup_transformations(config):
    
    # assert False, 'Check mask mult!'
    
    assert config.default.interpolate
    spatial_size_conf = tuple(config.default.interpolation_size)
    
    #if config.dataset.trim_background:
    keys=["image", "seg", "mask"]
    sep_k=["seg", "mask"]
    # else:
    #     keys=["image", "seg"]
    #     sep_k=["seg"]

    if config.opt.augmentation:
        rot_range = config.opt.rotation_range

        train_transf = Compose(
            [
                LoadImaged(keys=keys),
                EnsureChannelFirstd(keys=keys),
                # MaskIntensityd(keys=['image'], mask_key='mask'), 
                RandRotated(keys=keys, 
                            range_x=rot_range, 
                            range_y=rot_range, 
                            range_z=rot_range, 
                            prob=0.5),
                RandFlipd(keys=keys, prob=0.5, spatial_axis=0),
                Resized(keys=keys, spatial_size=spatial_size_conf),
                Spacingd(keys=sep_k, pixdim=1.0),
                ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0, channel_wise=True),
                masked_transform,
                EnsureTyped(keys=sep_k, dtype=torch.float),
            ]
        )

        val_transf = Compose(
            [
                LoadImaged(keys=keys),
                EnsureChannelFirstd(keys=keys),
                # MaskIntensityd(keys=['image'], mask_key='mask'),
                Resized(keys=keys, spatial_size=spatial_size_conf),
                Spacingd(keys=sep_k, pixdim=1.0),
                ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0, channel_wise=True),
                masked_transform,
                EnsureTyped(keys=sep_k, dtype=torch.float),
            ]
        )

    else:
        raise NotImplementedError
        
    return train_transf, val_transf



def setup_dataloaders(config):
    
    # load metadata: train-test split dict
    metadata_path = config.dataset.metadata_path
    split_dict = np.load(metadata_path, allow_pickle=True).item()   
    
    train_files, val_files = setup_datafiles(split_dict, config)
    train_transf, val_transf = setup_transformations(config)
    
    # training dataset
    train_ds = monai.data.Dataset(data=train_files, transform=train_transf)
    train_loader = DataLoader(
        train_ds,
        batch_size=config.opt.train_batch_size,
        shuffle=config.dataset.shuffle_train,
        num_workers=0,
        collate_fn=list_data_collate,
        pin_memory=torch.cuda.is_available(),
    )

    # validation dataset
    val_ds = monai.data.Dataset(data=val_files, transform=val_transf)
    val_loader = DataLoader(val_ds, 
                            batch_size=config.opt.val_batch_size, 
                            num_workers=0, 
                            collate_fn=list_data_collate,
                            shuffle=False # important not to shuffle, to ensure label correspondence
                            )
    
    return train_loader, val_loader