import os, subprocess
import numpy as np
from sklearn.metrics import cohen_kappa_score
from scipy import ndimage as nd
from scipy.ndimage import binary_opening
import nibabel as nib
import time
import multiprocessing
from sklearn.metrics import confusion_matrix
from nilearn import image
import pandas as pd
import contextlib
import warnings
warnings.filterwarnings("ignore")


def metric_clusters(input_scan, labels):
    """
    metric, that shows how many cluster ranged by size is needed to find lesion
    inputs:
    - input_scan: probabilistic input image (segmentation)
    - true labels
    output:
    - number of clusters
    """
    labels_scan = np.zeros_like(input_scan)
    brightness_scan = np.zeros_like(input_scan)
    brightness_scan_ = np.zeros_like(input_scan)
    bright_level = np.quantile(np.unique(input_scan),0.1)
    # perform morphological operations (dilation of the erosion of the input)
    morphed = nd.binary_opening(input_scan!=0, iterations=1)
    # label connected components
    morphed = nd.binary_fill_holes(morphed, structure=np.ones((5,5,5))).astype(int)
    pred_labels, _ = nd.label(morphed, structure=np.ones((3,3,3)))
    label_list = np.unique(pred_labels)
    num_elements_by_lesion = nd.labeled_comprehension(morphed, pred_labels, label_list, np.sum, float, 0)
    
    def quantile_(val):
        return np.quantile(val,0.75)
    
    def mean_bright(val):
        return np.mean(np.array([v for v in val if v > bright_level]))
    
    brightness_by_lesion = nd.labeled_comprehension(input_scan, pred_labels, label_list, quantile_, float, 0)
    
    brightness_by_lesion_ = nd.labeled_comprehension(input_scan, pred_labels, label_list, mean_bright, float, 0)
    brightness_by_lesion_ = np.nan_to_num(brightness_by_lesion_)
    if (num_elements_by_lesion[0]==0)&(len(num_elements_by_lesion)==1):
        return -2,-2,-2
    for l in range(len(num_elements_by_lesion)):
        if num_elements_by_lesion[l]>50:
        # assign voxels to output
            current_voxels = np.stack(np.where(pred_labels == l), axis=1)
            labels_scan[current_voxels[:,0], current_voxels[:,1], current_voxels[:,2]] = num_elements_by_lesion[l].astype(np.int)
            brightness_scan[current_voxels[:,0], current_voxels[:,1], current_voxels[:,2]] = brightness_by_lesion[l].astype(np.float)
            brightness_scan_[current_voxels[:,0], current_voxels[:,1], current_voxels[:,2]] = brightness_by_lesion_[l].astype(np.float)
    labels_scan = labels_scan*(labels>0.5)
    brightness_scan = brightness_scan*(labels>0.5)    
    brightness_scan_ = brightness_scan_*(labels>0.5)
    brightness_by_lesion = brightness_by_lesion[np.where((brightness_by_lesion!=0))]
    brightness_by_lesion_ = brightness_by_lesion_[np.where((brightness_by_lesion_!=0))]
    num_elements_by_lesion = np.sort(num_elements_by_lesion)[::-1]
    brightness_by_lesion = np.sort(brightness_by_lesion)[::-1]
    brightness_by_lesion_ = np.sort(brightness_by_lesion_)[::-1]
    n_cluster_size = np.where(num_elements_by_lesion==labels_scan.max())[0][0]+1
    try:
        n_cluster_bright = np.where(brightness_by_lesion==np.where(brightness_scan!=0,brightness_scan,-1000).max())[0][0]+1
    except:
        n_cluster_bright = -2
    try:
        n_cluster_bright_ = np.where(brightness_by_lesion_==np.where(brightness_scan_!=0,brightness_scan_,-1000).max())[0][0]+1
    except:
        n_cluster_bright_ = -2
    return n_cluster_size, n_cluster_bright, n_cluster_bright_


def metric_clusters_feature(feature):
    print(f'Start for sub {feature}')
    subs_ = pd.read_csv('./Tabels/FCNN.csv').iloc[:,0].values
    detection_table = np.zeros((len(subs_), 3))
    try:
        df_ = pd.read_csv(f'./Tabels/{feature}_clusters.csv')
        indeces_ = np.where((df_.iloc[:,2].values+df_.iloc[:,3].values)>0)[0]
        indeces = np.where((df_.iloc[:,2].values+df_.iloc[:,3].values)<=0)[0]
        subs = subs_[indeces]
        detection_table[indeces_,:] = df_.iloc[indeces_,1:]
        print('In first try')
    except:
        subs = subs_
        indeces = list(range(len(subs)))
        print('In first except')
        
    for j,sub in zip(indeces,subs):    
        try:
            label_path = f'/workspace/Features/preprocessed_data/label_bernaskoni/{sub}.nii.gz'
            prediction_path  = f'/workspace/Features/postprocessing/sub-{sub}/post-{feature}.nii.gz' 
            prediction = nib.load(prediction_path)
            prediction_data = prediction.get_fdata()
            label = (nib.load(label_path).get_fdata()>0.1).astype('uint8')
        except:
            detection_table[j,:] = [-1,-1,-1]
            continue
        print(f'Start for {sub} and {feature}')
        detection_table[j,:] = metric_clusters(prediction_data,label)
        print(f'Done for {sub} and {feature}')
        df = pd.DataFrame(detection_table, columns = ['size','brightness_q0.75','brightness_mean'], index=subs_)
        df.to_csv(f'/workspace/Tabels/{feature}_clusters.csv') 