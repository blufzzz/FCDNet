from comet_ml import Experiment
#from functions import *
import time
import multiprocessing
import numpy as np
import os
import re
import math
import pandas as pd
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import nibabel as nib
from nibabel import freesurfer
from nipype.interfaces.freesurfer import SurfaceTransform
from scipy.stats import zscore
from nilearn import image
from scipy.optimize import minimize_scalar
import matplotlib.image as mpimg
import warnings
import h5py
from tqdm import tqdm
from nilearn.plotting import plot_img
warnings.filterwarnings("ignore")


def our_metric(feature):
    print(f'start for {feature}')
    data_path = '/workspace/Features/prep_wf'
    subs_ = pd.read_csv('/workspace/Tabels/Features.csv').iloc[:,0].values
    
    detections_ = []
    detection_table = np.zeros((len(subs_), 2))
    df_metric = pd.DataFrame(columns=['subject', 'x', 'y', 'z', 'average_prediction', 'label_size', 'intersection_size'])
    n_of_subs = 0
    for j,sub in enumerate(subs_):
        if 'post' in feature:
            prediction_path  = f'/workspace/Features/postprocessing/sub-{sub}/{feature}.nii.gz'  
        else:
            if feature not in ['Thickness','Sulc','Curv']:
                prediction_path  = f'/workspace/Features/prep_wf/sub-{sub}/norm-{feature}.nii.gz'
            else:
                prediction_path  = f'/workspace/Features/prep_wf/sub-{sub}/{feature.low()}_mni.nii'    
        label_path = f'/workspace/Features/preprocessed_data/label_bernaskoni/{sub}.nii.gz'
        #try:
        prediction = nib.load(prediction_path)
        prediction_data = prediction.get_fdata()
        label = (nib.load(label_path).get_fdata()>0.1).astype('uint8')
        #except:
        #    detection_table[j,0] = -1
        #    detection_table[j,1] = -1
        #    print(f'Cannot open files for {sub}')
        #    continue
        n_of_subs += 1
        crops_df = pd.DataFrame(columns=['subject', 'x', 'y', 'z', 'average_prediction', 'label_size', 'intersection_size'])
        crop_size=np.array([20,20,20])#(np.array([64,64,64])/prediction.header.get_zooms()).astype(np.int64)
        i = 0 
        for x in range(0, prediction_data.shape[0]-crop_size[0]//2, crop_size[0]//2):
            for y in range(0, prediction_data.shape[1]-crop_size[1]//2, crop_size[1]//2):
                for z in range(0, prediction_data.shape[2]-crop_size[2]//2, crop_size[2]//2):

                    crop_pred = prediction_data[x: min(x+crop_size[0], prediction.shape[0]),
                                           y: min(y+crop_size[1], prediction.shape[1]),
                                           z: min(z+crop_size[2], prediction.shape[2]),]
                    crop_label = label[x: min(x+crop_size[0], prediction.shape[0]),
                                       y: min(y+crop_size[1], prediction.shape[1]),
                                       z: min(z+crop_size[2], prediction.shape[2]),]

                    crops_df.loc[i] = [sub, x, y, z, np.mean(crop_pred), label.sum(), crop_label.sum()]
                    i += 1
        if feature == 'Curv':
            top_10_crops_df = crops_df.sort_values(by='average_prediction', ascending=True)[:10]
            top_5_crops_df = crops_df.sort_values(by='average_prediction', ascending=True)[:5]
            
        else:
            top_10_crops_df = crops_df.sort_values(by='average_prediction', ascending=False)[:10]
            top_5_crops_df = crops_df.sort_values(by='average_prediction', ascending=False)[:5]
        if (top_10_crops_df.groupby('subject').intersection_size.max() > top_10_crops_df.groupby('subject').label_size.max()*0.5).any():
            detection_table[j,0] = 1        
        if (top_5_crops_df.groupby('subject').intersection_size.max() > top_5_crops_df.groupby('subject').label_size.max()*0.5).any():
            detection_table[j,1] = 1
        print(f'Done for {sub} and {feature}')
    df = pd.DataFrame(detection_table, columns = ['10 crops','5 crops'], index=subs_)
    df.to_csv(f'./Our_metric_{feature}.csv')