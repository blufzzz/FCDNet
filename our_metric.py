#from comet_ml import Experiment
from functions import *
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
    n_crops = 10
    data_path = f'/workspace/Features/prep_wf'
    #subs_ = pd.read_csv('/workspace/Tabels/Features.csv').iloc[:,0].values
    ls = os.listdir('/workspace/RawData/v2vNet/pred')
    ls = [x.split('.')[0] for x in ls if not x.startswith('.')]
    subs_ = ls
    detections_ = []
    detection_table = np.zeros((len(subs_), 1))
    k=0
    #for feature in ['Blurring T1','Blurring T2','Thickness','Sulc','Curv']:
    df_metric = pd.DataFrame(columns=['subject', 'x', 'y', 'z', 'average_prediction', 'label_size', 'intersection_size'])
    n_of_subs = 0
    for j,sub in enumerate(tqdm(subs_)):
        print(f'Start {feature} {sub}')
        if feature not in ['Thickness','Sulc','Curv']:
            prediction_path  = f'/workspace/RawData/Features/prep_wf/sub-{sub}/norm-{feature}.nii.gz'
        else:
            prediction_path  = f'/workspace/RawData/Features/prep_wf/sub-{sub}/{feature.lower()}_mni.nii'    
        label_path = f'/workspace/RawData/Features/preprocessed_data/label_bernaskoni/{sub}.nii.gz'
        if feature in ['v2v_t1-all_features']:
            prediction_path  = f'/workspace/RawData/Features/v2vNet/pred/{sub}.nii.gz'   
            label_path = f'/workspace/RawData/Features/v2vNet/label/{sub}.nii.gz'
        try:
            prediction = nib.load(prediction_path)
            prediction_data = prediction.get_fdata()
            label = (nib.load(label_path).get_fdata()>0.1).astype('uint8')
        except:
            print(f'Cannot open {prediction_path}')
            detection_table[j,k] = -1
            continue
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
            top_10_crops_df = crops_df.sort_values(by='average_prediction', ascending=True)[:n_crops]
        else:
            top_10_crops_df = crops_df.sort_values(by='average_prediction', ascending=False)[:n_crops]
        if (top_10_crops_df.groupby('subject').intersection_size.max() > top_10_crops_df.groupby('subject').label_size.max()*0.5).any():
            detection_table[j,k] = 1
        df_metric = pd.concat([df_metric, top_10_crops_df], ignore_index=True)
    detections = []
    for th in np.linspace(0, 1, 11):
        detections.append((df_metric.groupby('subject').intersection_size.max() > df_metric.groupby('subject').label_size.max()*th).sum())
    with open(f'/workspace/RawData/v2vNet/{feature}_.txt', 'w') as fp:
        fp.write("\n".join(str(item) for item in detections))
        

if __name__ == '__main__':
    #with metrics_experiment.train():
    start_mv = time.time()
    p = multiprocessing.Pool(processes=10)
    #features = ['Blurring_T1','Blurring_T2','Blurring_Flair','CR_Flair','CR_T2','Thickness','Sulc','Curv','Variance','Entropy', 'v2v_t1-all_features']
    features = ['v2v_t1-all_features']
    for f in features:
        print(f)
        p.apply_async(our_metric, [f])
        #metrics_(f)
    p.close()
    p.join()
    end_mv = time.time()
    print("Metrics: {} seconds".format(np.round(end_mv-start_mv, 1)))