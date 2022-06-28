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
    RandRotate90d, LabelToMaskd, RandFlipd, RandRotated, Spacingd, RandAffined
)
from monai.transforms.intensity.array import ScaleIntensity
import torch
from IPython.core.debugger import set_trace

BASE_DIR = '/workspace/RawData/Features'
OUTPUT_DIR = '/workspace/RawData/Features/BIDS'
TMP_DIR = '/workspace/Features/tmp'

FEATURES_LIST = ['image', 't2', 'flair', 'blurring-t1', 'blurring-Flair', 'cr-t2', 'cr-Flair', 'thickness', 'curv', 'sulc', 'variance']

def assign_feature_maps(sub, feature):
    global BASE_DIR
    global OUTPUT_DIR
    global TMP_DIR
    if feature == 'image':
        feature_map = os.path.join(BASE_DIR, f'prep_wf', f'sub-{sub}', f'sub-{sub}_t1_brain-final.nii.gz')
    elif feature == 't2':
        feature_map = os.path.join(BASE_DIR, f'prep_wf', f'sub-{sub}', f'sub-{sub}_t2_brain-final.nii.gz')
    elif feature == 'flair':
        feature_map = os.path.join(BASE_DIR, f'prep_wf', f'sub-{sub}', f'sub-{sub}_fl_brain-final.nii.gz')
    elif feature == 'blurring-t1':
        feature_map = os.path.join(BASE_DIR, f'prep_wf', f'sub-{sub}', f'Blurring_T1.nii.gz')
    elif feature == 'blurring-Flair':
        feature_map = os.path.join(BASE_DIR, f'prep_wf', f'sub-{sub}', f'Blurring_Flair.nii.gz')
    elif feature == 'cr-t2':
        feature_map = os.path.join(BASE_DIR, f'prep_wf', f'sub-{sub}', f'CR_T2.nii')
    elif feature == 'cr-Flair':
        feature_map = os.path.join(BASE_DIR, f'prep_wf', f'sub-{sub}', f'CR_Flair.nii')
    elif feature == 'thickness':
        feature_map = os.path.join(BASE_DIR, f'prep_wf', f'sub-{sub}', f'thickness_mni.nii')
    elif feature == 'curv':
        feature_map = os.path.join(BASE_DIR, f'prep_wf', f'sub-{sub}', f'curv_mni.nii')
    elif feature == 'sulc':
        feature_map = os.path.join(BASE_DIR, f'prep_wf', f'sub-{sub}', f'sulc_mni.nii')
    elif feature == 'variance':
        feature_map = os.path.join(BASE_DIR, f'preprocessed_data', 'var', f'sub-{sub}_var.nii.gz')
    elif feature == 'mask':
        feature_map = os.path.join(BASE_DIR, f'prep_wf', f'sub-{sub}', f'sub-{sub}_t1_brain-final_mask.nii.gz')
    return feature_map



def create_datafile(sub_list, feat_params):
    
    '''
    for each subject from `sub_list` 
    collects corresponding features from `feat_params`
    and segmentation mask
    '''

    files = []
    missing_files = defaultdict(list)

    for sub in sub_list:
        images_per_sub = dict()
        images_per_sub['image'] = []
        for feat in feat_params:
            map_path = assign_feature_maps(sub, feat)
            if os.path.isfile(map_path):
                images_per_sub['image'].append(map_path)
            else:
                missing_files['image'].append(map_path)
        
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
    '''
    
    train_list = split_dict.get('train')
    val_list = split_dict.get('test')
    
    images_list = []
    subject_list = []
    feat_params = config.dataset.features

    for i in os.listdir(OUTPUT_DIR):
        sub_ind = re.findall('-(.[a-zA-Z0-9]*|[0-9])', str(i))
        # if sub_ind and not any(x in sub_ind[0] for x in matches): 
        # subjects with 'n', 'G', 'NS', 'C' won't be included
        if sub_ind:
            subject_list.append(sub_ind[0])
    
    train_files, train_missing_files = create_datafile(train_list, feat_params)
    val_files, val_missing_files = create_datafile(val_list, feat_params)
    
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
    
    assert config.default.interpolate
    spatial_size_conf = tuple(config.default.interpolation_size)

    if config.opt.augmentation:
        rot_range = config.opt.rotation_range

        train_transf = Compose(
            [
                LoadImaged(keys=["image", "seg"]),
                EnsureChannelFirstd(keys=["image", "seg"]),
                RandRotated(keys=["image", "seg"], 
                            range_x=rot_range, 
                            range_y=rot_range, 
                            range_z=rot_range, 
                            prob=0.5),
                RandFlipd(keys=["image", "seg"], prob=0.5, spatial_axis=0),
                Resized(keys=["image", "seg"], spatial_size=spatial_size_conf, mode=('area', 'nearest')),
                Spacingd(keys=['seg'], pixdim=1.0),
                ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0, channel_wise=True),
                EnsureTyped(keys=["image", "seg"], dtype=torch.float),
            ]
        )

        val_transf = Compose(
            [
                LoadImaged(keys=["image", "seg"]),
                EnsureChannelFirstd(keys=["image", "seg"]),
                Resized(keys=["image", "seg"], spatial_size=spatial_size_conf, mode=('area', 'nearest')),
                Spacingd(keys=['seg'], pixdim=1.0),
                ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0, channel_wise=True),
                EnsureTyped(keys=["image", "seg"], dtype=torch.float),
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