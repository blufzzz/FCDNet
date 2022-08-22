import os 
import numpy as np
import nibabel as nib
from torch.nn import functional as F
import torch
from sklearn.metrics import confusion_matrix,accuracy_score


def calculate_metrics(data_resica, data_active):
    """
    Calculates matrics as functions of the threshold
    Returns: Maximum metrics and intesity, where maximum sum of metrics
    
    """
    # data_resica - pred
    # data_active - gt
    metrics = []
    if (data_resica==0).all():
        return 0,0,1,0,0
    
    for thresh in np.linspace(0,data_resica.max(),10):
        large_val = 0
        thresh_val = 0
        tn, fp, fn, tp = confusion_matrix((data_active > 0.5).reshape(-1), (data_resica > thresh).reshape(-1)).ravel()
        large_enough = int((np.where((data_resica > thresh),1,0)).sum() > 800)
        metrics.append([thresh, large_enough*tp/(tp+fp),large_enough*tp/(tp+fn),large_enough*tn/(tn+fp),large_enough*2*tp/(2*tp+fp+fn)])
        
    metrics = np.nan_to_num(np.array(metrics))
    
    ind = np.argmax(metrics[:,1]+metrics[:,2]+metrics[:,3]+metrics[:,4])
    intensity =metrics[:,0][ind]
    Precision = metrics[ind,1]
    Sensitivity = metrics[ind,2]
    Specificity = metrics[ind,3]
    Dice = metrics[ind,4]
    
    return Precision,Sensitivity,Specificity,Dice,intensity