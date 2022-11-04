import enum
import time
import os
import random
import multiprocessing
from pathlib import Path

import torch
#import torchvision
import torchio as tio
import torch.nn.functional as F

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

from torchio.transforms import HistogramStandardization
#from tqdm.auto import tqdm

BASE_DIR = '/workspace/RawData/Features'
def assign_feature_maps(sub, feature, norm=None):
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
        if norm:
            feature_map = os.path.join(BASE_DIR, f'prep_wf', f'sub-{sub}', f'norm-Blurring_T1.nii.gz')
        else:
            feature_map = os.path.join(BASE_DIR, f'prep_wf', f'sub-{sub}', f'Blurring_T1.nii.gz')
            
    elif feature == 'blurring-t2':
        if norm:
            feature_map = os.path.join(BASE_DIR, f'prep_wf', f'sub-{sub}', f'norm-Blurring_T2.nii.gz')
        else:
            feature_map = os.path.join(BASE_DIR, f'prep_wf', f'sub-{sub}', f'Blurring_T2.nii.gz')
            
    elif feature == 'blurring-Flair':
        if norm:
            feature_map = os.path.join(BASE_DIR, f'prep_wf', f'sub-{sub}', f'norm-Blurring_Flair.nii.gz')
        else:
            feature_map = os.path.join(BASE_DIR, f'prep_wf', f'sub-{sub}', f'Blurring_Flair.nii.gz')
            
    elif feature == 'cr-t2':
        if norm:
            feature_map = os.path.join(BASE_DIR, f'prep_wf', f'sub-{sub}', f'norm-CR_T2.nii.gz')
        else:
            feature_map = os.path.join(BASE_DIR, f'prep_wf', f'sub-{sub}', f'CR_T2.nii.gz')
            
    elif feature == 'cr-Flair':
        if norm:
            feature_map = os.path.join(BASE_DIR, f'prep_wf', f'sub-{sub}', f'norm-CR_Flair.nii.gz')
        else:
            feature_map = os.path.join(BASE_DIR, f'prep_wf', f'sub-{sub}', f'CR_Flair.nii.gz')
                
    elif feature == 'thickness':
        feature_map = os.path.join(BASE_DIR, f'prep_wf', f'sub-{sub}', f'thickness_mni.nii')
        
    elif feature == 'curv':
        feature_map = os.path.join(BASE_DIR, f'prep_wf', f'sub-{sub}', f'curv_mni.nii')
        
    elif feature == 'sulc':
        feature_map = os.path.join(BASE_DIR, f'prep_wf', f'sub-{sub}', f'sulc_mni.nii')
        
    elif feature == 'variance':
        if norm:
            feature_map = os.path.join(BASE_DIR, f'prep_wf', f'sub-{sub}', f'norm-Variance.nii.gz')
        else:
            feature_map = os.path.join(BASE_DIR, f'prep_wf', f'sub-{sub}', f'Variance.nii.gz')
            
    elif feature == 'entropy':
        if norm:
            feature_map = os.path.join(BASE_DIR, f'prep_wf', f'sub-{sub}', f'norm-Entropy.nii.gz')
        else:
            feature_map = os.path.join(BASE_DIR, f'prep_wf', f'sub-{sub}', f'Entropy.nii.gz')
            
    elif feature == 'border':
        feature_map = f'/workspace/borders/sub-{sub}_border.nii.gz'
            
    elif feature == 'GM':
        feature_map = os.path.join(BASE_DIR, f'prep_wf', f'sub-{sub}', f'c1sub-{sub}_space-MNI152NLin2009asym_T1w.nii')
        
    elif feature == 'WM':
        feature_map = os.path.join(BASE_DIR, f'prep_wf', f'sub-{sub}', f'c2sub-{sub}_space-MNI152NLin2009asym_T1w.nii')
    
    elif feature == 'CSF':
        feature_map = os.path.join(BASE_DIR, f'prep_wf', f'sub-{sub}', f'c3sub-{sub}_space-MNI152NLin2009asym_T1w.nii')
        
    elif feature == 'mask':
        feature_map = os.path.join(BASE_DIR, f'prep_wf', f'sub-{sub}', f'sub-{sub}_t1_brain-final_mask.nii.gz')
    
    return feature_map


#i=4
#metadata_path = "/workspace/folds_cv_nG.npy"
metadata_path = "/workspace/FCDNet/metadata/metadata_fcd_nG.npy"
split_dict = np.load(metadata_path, allow_pickle=True).item()
train_list = split_dict.get('train')
val_list = split_dict.get('test')
subs = np.concatenate((train_list, val_list))
Path(f'/workspace/FCDNet/landmarks').mkdir(exist_ok=True)

for feature in ['image', 't2', 'flair', 'blurring-t1', 'blurring-t2', 'blurring-Flair', 'cr-t2', 'cr-Flair', 'thickness', 'curv', 'sulc', 'variance', 'entropy']:
    mask_path = []
    for n in [True, False]:
        image_paths = []
        for sub in subs:
            image_paths.append(assign_feature_maps(sub, feature, norm=n))
            mask_paths.append(assign_feature_maps(sub, 'mask'))
        #subjects = []
        #for image_path in image_paths:
        #    subject = tio.Subject(
        #        mri=tio.ScalarImage(image_path),
        #    )
        #    subjects.append(subject)
        #dataset = tio.SubjectsDataset(subjects)
        if feature not in ['blurring-t1', 'blurring-t2', 'blurring-Flair', 'cr-t2', 'cr-Flair', 'variance', 'entropy']:
            landmarks_path = Path(f'/workspace/FCDNet/landmarks/{feature}_landmarks.npy')
        else:
            landmarks_path = Path(f'/workspace/FCDNet/landmarks/{feature}_{n}_landmarks.npy')
        mask_path = ''
        landmarks = (
            HistogramStandardization.train(image_paths, mask_path)
        )
        print(landmarks)
        np.save(landmarks_path, landmarks)
        
        if feature not in ['blurring-t1', 'blurring-t2', 'blurring-Flair', 'cr-t2', 'cr-Flair', 'variance', 'entropy']:
            break
