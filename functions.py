import os 
import numpy as np
#import re
#import math
#import pandas as pd
#import seaborn as sns
#from scipy.optimize import minimize_scalar
#import matplotlib.pyplot as plt
#import argparse
import nibabel as nib
#from nibabel import freesurfer
#from nilearn.plotting import plot_img
#from nipype.interfaces.freesurfer import SurfaceTransform
#from scipy.stats import zscore
from nilearn import image
#from scipy.optimize import minimize_scalar
#import matplotlib.image as mpimg
#from tqdm import tqdm
from torch.nn import functional as F
import torch
from sklearn.metrics import confusion_matrix,accuracy_score
from radiomics.glcm import RadiomicsGLCM
import SimpleITK as sitk
import warnings
warnings.filterwarnings("ignore")

BASE_DIR = '/workspace/RawData/Features/prep_wf'
MASK_B = nib.load('/workspace/RawData/Features/templates/MNI152_T1_1mm_brain_mask.nii.gz')
MASK_E = nib.load('/workspace/RawData/Features/templates/exclusive_mask_MNI1mm.nii.gz')
LABEL_PATH = '/workspace/RawData/Features/preprocessed_data/label_bernaskoni'
BAD_SUBS = ['55','n25','84']

def assign_feature_maps(sub, feature, norm):
    global BASE_DIR
    global OUTPUT_DIR
    global TMP_DIR
    if norm:
        if feature == 'Blurring T1':
            prediction_path  =  os.path.join(BASE_DIR,f'sub-{sub}/norm-Blurring_T1.nii.gz')
        elif feature == 'Blurring T2':
            prediction_path  = os.path.join(BASE_DIR, f'sub-{sub}/norm-Blurring_T2.nii.gz')    
        elif feature == 'Blurring Flair':
            prediction_path = os.path.join(BASE_DIR, f'sub-{sub}/norm-Blurring_Flair.nii.gz')
        elif feature == 'CR Flair':
            prediction_path = os.path.join(BASE_DIR,f'sub-{sub}/norm-CR_Flair.nii.gz')
        elif feature == 'CR T2':
            prediction_path = os.path.join(BASE_DIR,f'sub-{sub}/norm-CR_T2.nii.gz')
        elif feature == 'Entropy':
            prediction_path = os.path.join(BASE_DIR,f'sub-{sub}/norm-Entropy.nii.gz')
        elif feature == 'Var':
            prediction_path = os.path.join(BASE_DIR,f'sub-{sub}/norm-Variance.nii.gz')
        elif feature == 'Contrast':
            prediction_path = f'/workspace/RawData/Features/preprocessed_data/glcm_test/sub-{sub}/n_Contrast.nii'
        elif feature == 'Energy':
            prediction_path = f'/workspace/RawData/Features/preprocessed_data/glcm_test/sub-{sub}/n_Energy.nii'
        elif feature == 'Entropy_glcm':
            prediction_path = f'/workspace/RawData/Features/preprocessed_data/glcm_test/sub-{sub}/n_Alpha.nii'
        elif feature == 'Entropy5_glcm':
            prediction_path = f'/workspace/RawData/Features/preprocessed_data/glcm_test/sub-{sub}/n_Alpha5.nii'
        elif feature == 'Thickness':
            prediction_path = f'/workspace/RawData/Features/prep_wf/sub-{sub}/n_thickness_mni.nii'
        elif feature == 'Sulc':
            prediction_path = f'/workspace/RawData/Features/prep_wf/sub-{sub}/n_sulc_mni.nii'
        elif feature == 'Curv':
            prediction_path = f'/workspace/RawData/Features/prep_wf/sub-{sub}/n_curv_mni.nii'
    else:
        if feature == 'Blurring T1':
            prediction_path  =  os.path.join(BASE_DIR,f'sub-{sub}/Blurring_T1.nii.gz')
        elif feature == 'Blurring T2':
            prediction_path  = os.path.join(BASE_DIR, f'sub-{sub}/Blurring_T2.nii.gz')    
        elif feature == 'Blurring Flair':
            prediction_path = os.path.join(BASE_DIR, f'sub-{sub}/Blurring_Flair.nii.gz')
        elif feature == 'CR Flair':
            prediction_path = os.path.join(BASE_DIR,f'sub-{sub}/CR_Flair.nii.gz')
        elif feature == 'CR T2':
            prediction_path = os.path.join(BASE_DIR,f'sub-{sub}/CR_T2.nii.gz')
        elif feature == 'Entropy':
            prediction_path = os.path.join(BASE_DIR,f'sub-{sub}/Entropy.nii.gz')
        elif feature == 'Var':
            prediction_path = os.path.join(BASE_DIR,f'sub-{sub}/Variance.nii.gz')
        elif feature == 'Contrast':
            prediction_path = f'/workspace/RawData/Features/preprocessed_data/glcm_test/sub-{sub}/Contrast.nii'
        elif feature == 'Energy':
            prediction_path = f'/workspace/RawData/Features/preprocessed_data/glcm_test/sub-{sub}/Energy.nii'
        elif feature == 'Entropy_glcm':
            prediction_path = f'/workspace/RawData/Features/preprocessed_data/glcm_test/sub-{sub}/Alpha.nii'
        elif feature == 'Entropy5_glcm':
            prediction_path = f'/workspace/RawData/Features/preprocessed_data/glcm_test/sub-{sub}/Alpha5.nii'
        elif feature == 'Thickness':
            prediction_path = f'/workspace/RawData/Features/prep_wf/sub-{sub}/thickness_mni.nii'
        elif feature == 'Sulc':
            prediction_path = f'/workspace/RawData/Features/prep_wf/sub-{sub}/sulc_mni.nii'
        elif feature == 'Curv':
            prediction_path = f'/workspace/RawData/Features/prep_wf/sub-{sub}/curv_mni.nii'
    return prediction_path


def blurring(sub,data_path, GM_path, WM_path, type_data='T1', coef = None):
    '''
    data_path - str, path to T1, T2 or Flair brain
    GM_path, WM_path - str, path to GM and WM masks
    type_data - str, 'T1', 'T2' or 'Flair'
    '''
    global MASK_E
    global MASK_B
    if 'sub' in sub:
        sub = sub[4:]
    try:
        brain = nib.load(data_path)
        brain_data = brain.get_fdata()
    except:
        print(f'No {type_data} data for sub-{sub}')
        return None
    
    brain_mask_data = image.resample_to_img(MASK_B,brain).get_fdata()
    mask_e_data = image.resample_to_img(MASK_E,brain, interpolation='nearest').get_fdata()
    brain_mask_data = np.where((brain_mask_data>0.5)&(mask_e_data<0.5),1,0) # ????????????
    
    try:
        GM = nib.load(GM_path)
        GM_data = GM.get_fdata()
        WM = nib.load(WM_path)
        WM_data = WM.get_fdata()
    except:
        print(f'No GM or WM data for sub-{sub}')
        return None
    
    if type_data=='T1':
        if coef == None:
            coef = 0.5
        threshold1 = brain_data[np.where((GM_data > 0.5))].mean() + coef * brain_data[np.where((GM_data > 0.5))].std() 
        threshold2 = brain_data[np.where((WM_data > 0.5))].mean() - coef * brain_data[np.where((WM_data > 0.5))].std()
        print(brain_data[np.where((GM_data > 0.5))].mean(),brain_data[np.where((GM_data > 0.5))].std())
        print(brain_data[np.where((WM_data > 0.5))].mean(),brain_data[np.where((WM_data > 0.5))].std())
        print(threshold1,threshold2,coef)
    elif type_data=='T2':        
        if coef == None:
            coef = 0.02 
        threshold1 = brain_data[np.where((WM_data > 0.5))].mean() + coef * brain_data[np.where((WM_data > 0.5))].std()
        threshold2 = brain_data[np.where((GM_data > 0.5))].mean() - coef * brain_data[np.where((GM_data > 0.5))].std() 
    else:
        if coef == None:
            coef = 0.05 
        threshold1 = brain_data[np.where((WM_data > 0.5))].mean() + coef * brain_data[np.where((WM_data > 0.5))].std()
        threshold2 = brain_data[np.where((GM_data > 0.5))].mean() - coef * brain_data[np.where((GM_data > 0.5))].std() 
        
    if threshold2<threshold1:
        #while (threshold2<threshold1) and (coef>0.01):
        #    print(coef)
        #    if type_data=='T1':
        #        coef -= 0.05
        #        threshold1 = brain_data[np.where((GM_data > 0.5))].mean() + coef * brain_data[np.where((GM_data > 0.5))].std() 
        #        threshold2 = brain_data[np.where((WM_data > 0.5))].mean() - coef * brain_data[np.where((WM_data > 0.5))].std()
        #    else: 
        #        coef -= 0.005 
        #        threshold1 = brain_data[np.where((WM_data > 0.5))].mean() + coef * brain_data[np.where((WM_data > 0.5))].std()
        #        threshold2 = brain_data[np.where((GM_data > 0.5))].mean() - coef * brain_data[np.where((GM_data > 0.5))].std() 
        print('Problem with threshold!')
    brain_data = np.where((brain_data < threshold2)&(brain_data > threshold1)&(brain_mask_data>0.01), 1, 0)
    shape = brain_data.shape
    brain_data = torch.Tensor(brain_data.reshape(1,1,shape[0],shape[1],shape[2]))
    brain_data = F.conv3d(brain_data ,torch.ones(1,1,5,5,5), None, 1, 2).reshape(shape[0],shape[1],shape[2]).numpy()
    brain_data = brain_data*np.where(brain_mask_data>0.01, 1, 0)
    brain_data = nib.Nifti1Image(brain_data, brain.affine)
    nib.save(brain_data, assign_feature_maps(sub, f'Blurring {type_data}', norm=False))
    return None
    
    
def CR(sub, data_path, GM_path, type_data='Flair', params = [1, 15, 5]):
    '''
    data_path - str, path to T2 or Flair brain
    GM_path - str, path to GM masks
    type_data - str, 'T2' or 'Flair'
    params - list of 3 numbers (size,num,m), where 2*size^3 - volume of window, num-m the britest voxels taken for summation
    '''
    global MASK_B
    global MASK_E
    if 'sub-' in sub:
        sub = sub[4:]
    size,num,m = params
    try:
        GM = nib.load(GM_path).get_fdata()    
        data = nib.load(data_path)
    except:
        print('No data')
        return None
    
    mask_data = image.resample_to_img(MASK_E,data, interpolation='nearest').get_fdata()

    brain_mask_data = image.resample_to_img(MASK_B,data, interpolation='nearest').get_fdata()
    brain_data = data.get_fdata()
    new_feature = np.zeros(brain_data.shape)

    for i in np.array(np.where((GM>0.6)&(mask_data<0.5))).T:
        if (i>size-1).all() & (i+size<brain_data.shape).all():
            square = brain_data[i[0]-size:i[0]+size+1,i[1]-size:i[1]+size+1,i[2]-size:i[2]+size+1]  
        else:
            continue
        #x = np.unique(square.reshape(-1))
        x = np.sort(square.reshape(-1))
        new_feature[i[0],i[1],i[2]] = x[x.shape[0]-num-m:-m].sum()
    new_feature = new_feature/new_feature.max()
    new_feature = new_feature * np.where(brain_mask_data>0.01,1,0)
    img = nib.Nifti1Image(new_feature, data.affine)
    nib.save(img, assign_feature_maps(sub, f'CR {type_data}', norm=False))   
    
    return None

def variance(sub, data_path, GM_path, size = 4):
    '''
    data_path - str, path to T2 or Flair brain
    GM_path - str, path to GM masks
    type_data - str, 'T2' or 'Flair'
    params - list of 3 numbers (size,num,m), where 2*size^3 - volume of window, num-m the britest voxels taken for summation
    '''
    global MASK_B
    global MASK_E
    if 'sub-' in sub:
        sub = sub[4:]
    try:
        GM = nib.load(GM_path).get_fdata()    
        data = nib.load(data_path)
    except:
        print('No data')
        return None
    
    mask_data = image.resample_to_img(MASK_E,data, interpolation='nearest').get_fdata()

    brain_mask_data = image.resample_to_img(MASK_B,data, interpolation='nearest').get_fdata()
    brain_data = data.get_fdata()
    new_feature = np.zeros(brain_data.shape)
    for i in np.array(np.where((GM>0.2)&(mask_data<0.5))).T:
        if (i>size-1).all() & (i+size<brain_data.shape).all():
            square = brain_data[i[0]-size:i[0]+size+1,i[1]-size:i[1]+size+1,i[2]-size:i[2]+size+1]  
        else:
            continue
        new_feature[i[0],i[1],i[2]] = square.std(ddof=1)
    new_feature = new_feature/new_feature.max()
    new_feature = new_feature * np.where(brain_mask_data>0.01,1,0)
    img = nib.Nifti1Image(new_feature, data.affine)
    nib.save(img, assign_feature_maps(sub, f'Var', norm=False))   
    
    return None


def kurtosis_(sub, data_path, GM_path, size = 4, bias = False):
    '''
    data_path - str, path to T2 or Flair brain
    GM_path - str, path to GM masks
    type_data - str, 'T2' or 'Flair'
    params - list of 3 numbers (size,num,m), where 2*size^3 - volume of window, num-m the britest voxels taken for summation
    '''
    global MASK_B
    global MASK_E
    if 'sub-' in sub:
        sub = sub[4:]
    try:
        GM = nib.load(GM_path).get_fdata()    
        data = nib.load(data_path)
    except:
        print('No data')
        return None
    
    mask_data = image.resample_to_img(MASK_E,data, interpolation='nearest').get_fdata()

    brain_mask_data = image.resample_to_img(MASK_B,data, interpolation='nearest').get_fdata()
    brain_data = data.get_fdata()
    new_feature = np.zeros(brain_data.shape)
    for i in np.array(np.where((GM>0.7)&(mask_data<0.5))).T:
        if (i>size-1).all() & (i+size<brain_data.shape).all():
            square = brain_data[i[0]-size:i[0]+size+1,i[1]-size:i[1]+size+1,i[2]-size:i[2]+size+1]  
        else:
            continue
        new_feature[i[0],i[1],i[2]] = kurtosis(square.reshape((size*2+1)**3),fisher=False,bias=bias)
    new_feature = new_feature/new_feature.max()
    new_feature = new_feature * np.where(brain_mask_data>0.01,1,0)
    img = nib.Nifti1Image(new_feature, data.affine)
    nib.save(img, assign_feature_maps(sub, f'Kurtosis', norm=False))   
    
    return None


def skewness_(sub, data_path, GM_path, size = 4, bias = False):
    '''
    data_path - str, path to T2 or Flair brain
    GM_path - str, path to GM masks
    type_data - str, 'T2' or 'Flair'
    params - list of 3 numbers (size,num,m), where 2*size^3 - volume of window, num-m the britest voxels taken for summation
    '''
    global MASK_B
    global MASK_E
    if 'sub-' in sub:
        sub = sub[4:]
    try:
        GM = nib.load(GM_path).get_fdata()    
        data = nib.load(data_path)
    except:
        print('No data')
        return None
    
    mask_data = image.resample_to_img(MASK_E,data, interpolation='nearest').get_fdata()

    brain_mask_data = image.resample_to_img(MASK_B,data, interpolation='nearest').get_fdata()
    brain_data = data.get_fdata()
    new_feature = np.zeros(brain_data.shape)
    for i in np.array(np.where((GM>0.7)&(mask_data<0.5))).T:
        if (i>size-1).all() & (i+size<brain_data.shape).all():
            square = brain_data[i[0]-size:i[0]+size+1,i[1]-size:i[1]+size+1,i[2]-size:i[2]+size+1]  
        else:
            continue
        new_feature[i[0],i[1],i[2]] = skew(square.reshape((size*2+1)**3),bias=bias)
    new_feature = new_feature/new_feature.max()
    new_feature = new_feature * np.where(brain_mask_data>0.01,1,0)
    img = nib.Nifti1Image(new_feature, data.affine)
    nib.save(img, assign_feature_maps(sub, f'Skewness', norm=False))   
    
    return None


def x_log2_x(x):
    """ Return x * log2(x) and 0 if x is 0."""
    results = x * np.log2(x)
    if np.size(x) == 1:
        if np.isclose(x, 0.0):
            results = 0.0
    else:
        results[np.isclose(x, 0.0)] = 0.0
    return results

def renyi_entropy(alpha, X):
    assert alpha >= 0, "Error: renyi_entropy only accepts values of alpha >= 0, but alpha = {}.".format(alpha)  # DEBUG
    if np.isinf(alpha):
        return - np.log2(np.max(X))
    elif np.isclose(alpha, 0):
        return np.log2(len(X))
    elif np.isclose(alpha, 1):
        return - np.sum(x_log2_x(X))
    else:
        return (1.0 / (1.0 - alpha)) * np.log2(np.sum(X ** alpha))
    

def entropy(sub, data_path, GM_path, params = [5,7]):
    '''
    
    '''
    size,alpha = params
    global MASK_E
    if 'sub-' in sub:
        sub = sub[4:]
    try:
        GM = nib.load(GM_path).get_fdata()    
        data = nib.load(data_path)
    except:
        print('No data')
        return None
    brain_data = data.get_fdata()
    mask_data = image.resample_to_img(MASK_E,data, interpolation='nearest').get_fdata()
    new_feature = np.zeros(brain_data.shape)
    for i in np.array(np.where((GM>0.6)&(mask_data<0.5))).T:
        if (i>size-1).all() & (i+size<brain_data.shape).all():
            square = brain_data[i[0]-size:i[0]+size+1,i[1]-size:i[1]+size+1,i[2]-size:i[2]+size+1]
        else:
            continue
        #x = np.unique(square.reshape(-1))
        x = square.reshape(-1)
        new_feature[i[0],i[1],i[2]] = renyi_entropy(alpha, x/np.sum(x))
    new_feature = np.nan_to_num(new_feature)
    new_feature = new_feature/new_feature.max()
    img = nib.Nifti1Image(new_feature, data.affine)    
    nib.save(img, assign_feature_maps(sub, f'Entropy', norm=False)) 
    return None


def energy_contrast(sub, data_path, GM_path, size=5):
    try:
        GM = nib.load(GM_path).get_fdata()
        flair = nib.load(data_path)
    except:
        print('No data')
    flair_data = flair.get_fdata()
    energy_feature = np.zeros(flair_data.shape)
    contrast_feature = np.zeros(flair_data.shape)
    
    for i in np.array(np.where(GM>0.55)).T:
        if (i>size-1).all() & (i+size<flair_data.shape).all():
            square = flair_data[i[0]-size:i[0]+size+1,i[1]-size:i[1]+size+1,i[2]-size:i[2]+size+1]
            mask =  np.where(GM[i[0]-size:i[0]+size+1,i[1]-size:i[1]+size+1,i[2]-size:i[2]+size+1]>0.1,1,0)
        else:
            continue
        GLSMs = RadiomicsGLCM(sitk.GetImageFromArray(square),sitk.GetImageFromArray(mask), distances=[1,2,3,4])
        GLSMs._initCalculation()
        matrix = GLSMs.P_glcm
        new_feature_outmethod = GLSMs.getJointEnergyFeatureValue
        matrix = matrix.sum(axis=3).reshape(matrix.shape[1],matrix.shape[1])
        matrix = matrix / matrix.sum()
        g = matrix.shape[0]
        for k in range(int(g/2),g):
            for j in range(int(g/2),g):
                energy_feature[i[0],i[1],i[2]] += matrix[k,j]**2
        for k in range(g):
            for j in range(g):
                contrast_feature[i[0],i[1],i[2]] += (k-j)**2 * matrix[k,j] 
        #        correlation_feature[i[0],i[1],i[2]] += (k-np.mean(matrix[k,:]))*(j-np.mean(matrix[:,j]))*(matrix[k,j])/np.std(matrix[k,:])/np.std(matrix[:,j])
        #        shade_feature[i[0],i[1],i[2]] += (k-np.mean(matrix[k,:])+j-np.mean(matrix[:,j]))**3*(matrix[k,j])
        #        if matrix[k,j] != 0:
        #            entropy_feature[i[0],i[1],i[2]] += matrix[k,j] * np.log(matrix[k,j])
                
    img_energy = nib.Nifti1Image(energy_feature, flair.affine)
    img_contrast = nib.Nifti1Image(contrast_feature, flair.affine)
    nib.save(img_energy, assign_feature_maps(sub, f'Energy', norm=False))    
    nib.save(img_contrast, assign_feature_maps(sub, f'Contrast', norm=False)) 


def quantiles(qs,feature,type_data):
    '''
    qs - list, of pairs q1, q2, where q1 - large quantile (ex. 0.95), q2 - small quantile (ex. 0.05)
    feature - str, from ['Blurring T1','Blurring T2','Blurring Flair','Thickness','Sulc','Curv','CR Flair','CR T2','Entropy']
    type_data = str, from ['T1', 'T2', 'Flair']
    '''
    global BASE_DIR 
    global MASK_B
    global BAD_SUBS
    DIR = '/workspace/RawData/Features/preprocessed_data'
    for pos in ['right','left']:
        features = 0
        for i,sub in enumerate(os.listdir(os.path.join(DIR,pos,type_data))):
            if ('sub' not in sub) or ('nii' not in sub):
                continue
            if sub[4:-4] in BAD_SUBS:
                continue
            half = nib.load(os.path.join(DIR,pos,type_data,sub))
            try:
                feature_ = nib.load(assign_feature_maps(sub[4:-4], feature, norm=False))
                half = np.where(image.resample_to_img(half,feature_).get_fdata()>0.01,1,0)
            except:
                print('no data')
                continue
            feature_data = feature_.get_fdata()
            #if feature not in ['Thickness','Sulc', 'Curv']:
            if feature_data.shape!=(197, 233, 189):
                print('wrong shape')
                continue
            feature_data = feature_data*half
            if features == 0:
                features = feature_
                features_data = feature_data.reshape(1,feature_data.shape[0],feature_data.shape[1],feature_data.shape[2])
            else:
                #if feature in ['Thickness','Sulc', 'Curv']:
                #    data_half = nib.Nifti1Image(feature_data, feature_.affine)
                #    feature_data = image.resample_to_img(data_half,features)
                #    feature_data = feature_data.get_fdata()
                if np.isnan(feature_data).any()==True:
                    print('none')
                    continue
                feature_data = feature_data.reshape(1,feature_data.shape[0],feature_data.shape[1],feature_data.shape[2])
                features_data = np.concatenate([features_data,feature_data], axis=0) 
        for q1,q2 in qs:
            features_data_l = np.quantile(features_data,q1,axis=0)
            features_data_s = np.quantile(features_data,q2,axis=0)
            features_data_l = nib.Nifti1Image(features_data_l, features.affine)
            features_data_s = nib.Nifti1Image(features_data_s, features.affine)
            nib.save(features_data_l, os.path.join(DIR,pos,feature.replace(" ", ""),f'q{str(q1)[2:]}.nii.gz'))
            nib.save(features_data_s, os.path.join(DIR,pos,feature.replace(" ", ""),f'q{str(q2)[2:]}.nii.gz'))
        features_data_mean = features_data.mean(axis=0)
        features_data_mean = nib.Nifti1Image(features_data_mean, features.affine)
        nib.save(features_data_mean, os.path.join(DIR,pos,feature.replace(" ", ""),f'mean.nii.gz'))
        np.save(os.path.join(DIR,pos,feature.replace(" ", ""),f'dist.npy'),features_data)
        
    for phase in ['mean']+[f'q{str(q)[2:]}' for q_ls in qs for q in q_ls]:    
        left = nib.load(os.path.join(DIR,'left',feature.replace(" ", ""),f'{phase}.nii.gz'))
        right = nib.load(os.path.join(DIR,'right',feature.replace(" ", ""),f'{phase}.nii.gz'))
        #if feature in ['Thickness','Sulc', 'Curv']:
        #    right_data = image.resample_to_img(right,left).get_fdata() 
        #else:
        right_data =  right.get_fdata()
        left_data = left.get_fdata()
        #print(phase,': mean_r =', right_data.mean(),', mean_l =', left_data.mean())
        mask_b_data = image.resample_to_img(MASK_B,left).get_fdata() 
        unique_v, unique_c = np.unique(np.where(((left_data+right_data)<1e-10)&(mask_b_data>0.))[0], return_counts=True)
        ind = unique_v[np.argmax(unique_c)]
        left_data[ind,:,:] = left_data[ind-1,:,:]
        whole = nib.Nifti1Image(left_data+right_data,left.affine)
        nib.save(whole,os.path.join(DIR,'quantiles',feature.replace(" ", ""),f'{phase}.nii.gz'))
    
        
def calculate_precision(data_resica, data_active):
    """
    Calculates precision as function of the threshold, save the Precision(threshold) as fig_name (if such is given)
    The input args are described above
    Returns: Maximum precision and intesity, where maximum precision
    
    """
    precision = []
    for thresh in np.linspace(0,data_resica.max(),50):
        y = (((data_active > 0.5) & (np.where(data_resica > thresh,1,0))).sum() /
                ((np.where(data_resica > thresh,1,0)).sum())) * int((np.where(data_resica > thresh,1,0)).sum() > 800)
        precision.append([thresh, y])

    precision = np.array(precision)


    def f(thresh):
        return -(((data_active > 0.5) & (np.where(data_resica > thresh,1,0))).sum() /
                ((np.where(data_resica > thresh,1,0)).sum())) * int((np.where(data_resica > thresh,1,0)).sum() > 800)
    Precision = max(precision[:,1])
    intensity =precision[:,0][np.argmax(precision[:,1])]
    return intensity, Precision
        
def calculate_metrics(data_resica, data_active):
    """
    Calculates matrics as functions of the threshold
    Returns: Maximum metrics and intesity, where maximum sum of metrics
    
    """
    metrics = []
    if (data_resica==0).all():
        return 0,0,1,0,0
    
    for thresh in np.linspace(0,data_resica.max(),10):
        tn, fp, fn, tp = confusion_matrix((data_active > 0.5).reshape(-1), (data_resica > thresh).reshape(-1)).ravel()
        large_enough = int((np.where((data_resica > thresh),1,0)).sum() > 800)
        #metrics.append([thresh, large_enough*tp/(tp+fp),large_enough*tp/(tp+fn),large_enough*tn/(tn+fp),
        #                large_enough*(tp+tn)/(tp+tn+fp+fn),large_enough*fp/(fn+fp),large_enough*2*tp/(2*tp+fp+fn),large_enough*tp/(tp+fp+fn)])
        metrics.append([thresh, large_enough*tp/(tp+fp),large_enough*tp/(tp+fn),large_enough*tn/(tn+fp),large_enough*2*tp/(2*tp+fp+fn)])
        
    metrics = np.nan_to_num(np.array(metrics))
    
    ind = np.argmax(metrics[:,1]+metrics[:,2]+metrics[:,3]+metrics[:,4])
    intensity =metrics[:,0][ind]
    Precision = metrics[ind,1]
    Sensitivity = metrics[ind,2]
    Specificity = metrics[ind,3]
    Dice = metrics[ind,4]
    
    return Precision,Sensitivity,Specificity,Dice,intensity



def normalization(q1,q2,sub,feature,save=False,metric='all'):
    '''
    q1, q2 - where q1 - large quantile (ex. 0.95), q2 - small quantile (ex. 0.05)
    feature - str, from ['Blurring T1','Blurring T2','Blurring Flair','Thickness','Sulc','Curv','CR Flair','CR T2','Entropy']
    '''
    global LABEL_PATH  
    global BASE_DIR
    global MASK_B
    global MASK_E
    
    if q1==0.5:
        Cl = nib.load(os.path.join('/workspace/RawData/Features/preprocessed_data/quantiles',feature.replace(" ", ""),f'mean.nii.gz'))
    else:
        Cl = nib.load(os.path.join('/workspace/RawData/Features/preprocessed_data/quantiles',feature.replace(" ", ""),f'q{str(q1)[2:]}.nii.gz'))
        Cs = nib.load(os.path.join('/workspace/RawData/Features/preprocessed_data/quantiles',feature.replace(" ", ""),f'q{str(q2)[2:]}.nii.gz'))
    if 'sub-' in sub:
        sub = sub[4:]
    feature_map = nib.load(assign_feature_maps(sub, feature, norm=False))
    brain_mask_data = image.resample_to_img(MASK_B,feature_map, interpolation='nearest').get_fdata()
    mask_e_data = image.resample_to_img(MASK_E,feature_map, interpolation='nearest').get_fdata()
    brain_mask_data = np.where((brain_mask_data>0.5)&(mask_e_data<0.5),1,0) 
    label = nib.load(os.path.join(LABEL_PATH,f'{sub}.nii.gz'))
    #if feature in ['Thickness','Sulc', 'Curv']: 
    #    label_data = image.resample_to_img(label, feature_map, interpolation='nearest').get_fdata() 
    #    Cl_ = image.resample_to_img(Cl,feature_map).get_fdata()
    #else:
    label_data = label.get_fdata()
    Cl_ = Cl.get_fdata()
    if q1!=0.5:
        #if feature in ['Thickness','Sulc', 'Curv']:
        #    Cs_ = image.resample_to_img(Cs,feature_map).get_fdata()
        #else:
        Cs_ = Cs.get_fdata()
    feature_data = feature_map.get_fdata()
    if feature_data.shape!=(197, 233, 189):
        return None
    if q1==0.5:
        feature_data_ = abs(feature_data-Cl_)*brain_mask_data
    else:
        feature_data_ = np.zeros(feature_data.shape)
        feature_data_ = np.where(feature_data>Cl_,feature_data-Cl_,feature_data_)
        if q1!=0.5:
            feature_data_ = np.where(feature_data<Cs_,Cs_-feature_data,feature_data_)
            feature_data_ = np.where((feature_data<Cl_)&(feature_data>Cs_),0,feature_data_)
        if q1==0.5:
            feature_data_ = abs(feature_data-Cl_)
    feature_data_ = feature_data_*brain_mask_data
    feature_data = nib.Nifti1Image(feature_data_, feature_map.affine)
    if save:
        nib.save(feature_data, assign_feature_maps(sub, feature, norm=True))
    if metric=='all':
        #Precision,Sensitivity,Specificity,Accuracy,Precision_,Dice,IOU,_ = calculate_metrics(feature_data_, label_data)
        Precision,Sensitivity,Specificity,Dice,_ = calculate_metrics(feature_data_, label_data)
    elif metric=='precision':
        _,Precision = calculate_precision(feature_data_, label_data) 
        return Precision
    #return Precision,Sensitivity,Specificity,Accuracy,Precision_,Dice,IOU
    return Precision,Sensitivity,Specificity,Dice