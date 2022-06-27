# https://github.com/frankkramer-lab/MIScnn
import numpy as np
import torch
from IPython.core.debugger import set_trace
from torch import nn

def bce_weighted(delta=0.5):
    def loss_function(y_pred, y_true):

        epsilon=1e-10

        device = targets.device
        dtype = logits.dtype

        logits = torch.clip(logits, epsilon, 1. - epsilon)
        logits, targets = binary_to_multiclass(logits, targets)

        weights = torch.ones(targets.shape, dtype=dtype).to(device) # [bs,2,d,d,d]
        weights[:,0,...] = weights[:,0,...]*config.opt.bce_neg_weight # background
        weights[:,1,...] = weights[:,1,...]*config.opt.bce_pos_weight # foreground

        cross_entropy = -targets * weights * torch.log(logits)
        return cross_entropy.sum(1).mean() # sum neg and pos pixels and average.

    return loss_function


def binary_to_multiclass(y_pred, y_true):
    '''
    Maps 1-channel tensors with foreground class to 
    2-channel tensors with foreground class with index 1
    and background class with index 0
    '''
    # stack background and foreground
    y_pred_ = torch.cat([1-y_pred, y_pred],1) 
    y_true_ = torch.cat([1-y_true, y_true],1)

    return y_pred_, y_true_


# Helper function to enable loss function to be flexibly used for both 2D or 3D image segmentation
def identify_dim(shape):
    # Three dimensional: [bs,C,H,W,D]
    if len(shape) == 5 : return [2,3,4]
    # Two dimensional: [bs,C,H,W]
    elif len(shape) == 4 : return [2,3]
    # Exception - Unknown
    else : raise ValueError('Metric: Shape of tensor is neither 2D or 3D.')


################################
#       Dice coefficient       #
################################
def dice_coefficient_weighted(delta = 0.5, smooth = 0.000001):
    """The Dice similarity coefficient, also known as the Sørensen–Dice index or simply Dice coefficient, 
    is a statistical tool which measures the similarity between two sets of data.
    Parameters
    ----------
    delta : float, optional
        controls weight given to false positive and false negatives, by default 0.5
    smooth : float, optional
        smoothing constant to prevent division by zero errors, by default 0.000001
    """
    def loss_function(y_pred, y_true):
        dim = identify_dim(y_true.shape)
        # Calculate true positives (tp), false negatives (fn) and false positives (fp)   
        tp = torch.sum(y_true * y_pred, dim=dim)
        fn = torch.sum(y_true * (1-y_pred), dim=dim)
        fp = torch.sum((1-y_true) * y_pred, dim=dim)
        dice_class = (tp + smooth)/(tp + delta*fn + (1-delta)*fp + smooth)
        # Average class scores
        dice = torch.mean(dice_class)

        return dice

    return loss_function


################################
#         Tversky loss         #
################################
def tversky_loss(delta = 0.7, smooth = 0.000001):
    """Tversky loss function for image segmentation using 3D fully convolutional deep networks
    Link: https://arxiv.org/abs/1706.05721
    Parameters
    ----------
    delta : float, optional
        controls weight given to false positive and false negatives, by default 0.7
    smooth : float, optional
        smoothing constant to prevent division by zero errors, by default 0.000001
    """

    def loss_function(y_pred, y_true):
        dim = identify_dim(y_true.shape)
        # Calculate true positives (tp), false negatives (fn) and false positives (fp)   
        tp = torch.sum(y_true * y_pred, dim=dim)
        fn = torch.sum(y_true * (1-y_pred), dim=dim)
        fp = torch.sum((1-y_true) * y_pred, dim=dim)
        tversky_class = (tp + smooth)/(tp + delta*fn + (1-delta)*fp + smooth)
        # Average class scores
        tversky_loss = torch.mean(1-tversky_class)

        return tversky_loss

    return loss_function


################################
#      Focal Tversky loss      #
################################
def focal_tversky_loss(delta=0.7, gamma=0.7, smooth=0.000001):
    """A Novel Focal Tversky loss function with improved Attention U-Net for lesion segmentation
    Link: https://arxiv.org/abs/1810.07842
    Parameters
    ----------
    gamma : float, optional
        focal parameter controls degree of down-weighting of easy examples, by default 0.75
    """
    def loss_function(y_pred, y_true):
        # Clip values to prevent division by zero error
        epsilon = 1e-10
        y_pred = torch.clip(y_pred, epsilon, 1. - epsilon) 
        dim = identify_dim(y_true.shape)
        # Calculate true positives (tp), false negatives (fn) and false positives (fp)     
        tp = torch.sum(y_true * y_pred, dim=dim)
        fn = torch.sum(y_true * (1-y_pred), dim=dim)
        fp = torch.sum((1-y_true) * y_pred, dim=dim)
        tversky_class = (tp + smooth)/(tp + delta*fn + (1-delta)*fp + smooth)
        # Average class scores
        focal_tversky_loss = torch.mean(torch.pow((1-tversky_class), gamma))
	
        return focal_tversky_loss

    return loss_function


#################################
# Symmetric Focal Tversky loss  #
#################################
def symmetric_focal_tversky_loss(delta=0.7, gamma=0.75):
    """This is the implementation for binary segmentation.
    Parameters
    ----------
    delta : float, optional
        controls weight given to false positive and false negatives, by default 0.7
    gamma : float, optional
        focal parameter controls degree of down-weighting of easy examples, by default 0.75
    """
    def loss_function(y_pred, y_true):

        epsilon = 1e-10
        # Clip values to prevent division by zero error
        y_pred = torch.clip(y_pred, epsilon, 1. - epsilon)

        y_pred, y_true = binary_to_multiclass(y_pred, y_true)

        dim = identify_dim(y_true.shape)

        # Calculate true positives (tp), false negatives (fn) and false positives (fp)     
        tp = torch.sum(y_true * y_pred, dim=dim)
        fn = torch.sum(y_true * (1-y_pred), dim=dim)
        fp = torch.sum((1-y_true) * y_pred, dim=dim)
        dice_class = (tp + epsilon)/(tp + delta*fn + (1-delta)*fp + epsilon)

        #calculate losses separately for each class, enhancing both classes
        back_dice = (1-dice_class[:,0]) 
        fore_dice = (1-dice_class[:,1]) 
        
        if gamma > 0:
            back_dice = back_dice*torch.pow(1-dice_class[:,0], gamma) 
            fore_dice = fore_dice*torch.pow(1-dice_class[:,1], gamma) 

        # Average class scores
        loss = torch.mean(torch.stack([back_dice, fore_dice], dim=-1))

        return loss

    return loss_function



################################
#          Focal loss          #
################################
def focal_loss(delta=0.9, gamma=1.):
    """Focal loss is used to address the issue of the class imbalance problem. 
    A modulation term applied to the Cross-Entropy loss function.
    Parameters
    ----------
    alpha : float, optional
        controls relative weight of false positives and false negatives. Beta > 0.5 penalises false negatives more than false positives, by default None
    gamma : float, optional
        focal parameter controls degree of down-weighting of easy examples, by default 2.
    """
    def loss_function(y_pred, y_true):

        y_pred, y_true = binary_to_multiclass(y_pred, y_true)

        dim = identify_dim(y_true.shape)
        # Clip values to prevent division by zero error
        epsilon = 1e-10
        y_pred = torch.clip(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * torch.log(y_pred)
        focal_loss = torch.pow(1-y_pred, gamma)*cross_entropy # [bs,1,d,d,d]
        focal_loss = torch.mean(torch.sum(focal_loss, dim=[-1,-2,-3]))
        return focal_loss
    return loss_function


################################
#       Symmetric Focal loss   #
################################
def symmetric_focal_loss(delta=0.7, gamma=2.):
    """
    Parameters
    ----------
    delta : float, optional
        controls weight given to false positive and false negatives, 
        by default 0.7
    gamma : float, optional
        Focal Tversky loss' focal parameter controls degree of down-weighting of easy examples, 
        by default 2.0
    """
    def loss_function(y_pred, y_true):

        y_pred, y_true = binary_to_multiclass(y_pred, y_true)

        dim = identify_dim(y_true.shape)  

        epsilon = 1e-5
        y_pred = torch.clip(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * torch.log(y_pred) # non-zero only where y_true non-zero

        # calculate losses separately for each class
        back_ce = (1 - delta)*cross_entropy[:,0]
        fore_ce = delta * cross_entropy[:,1]

        if gamma > 0:
            back_ce = back_ce * torch.pow(1 - y_pred[:,0], gamma) 
            fore_ce = fore_ce * torch.pow(1 - y_pred[:,1], gamma)

        n_back = y_true[:,0].sum()
        n_fore = y_true[:,1].sum()

        loss = 0
        if n_back > 0:
            loss += back_ce.sum()/n_back
        if n_fore > 0:
            loss += fore_ce.sum()/n_fore

        return loss

    return loss_function



#################################
#      Unified Focal loss       #
#################################
def unified_focal_loss(weight=0.5, delta=0.6, gamma=0.5):
    """The Unified Focal loss is a new compound loss function that unifies Dice-based and cross entropy-based loss functions into a single framework.
    Parameters
    ----------
    weight : float, optional
        represents lambda parameter and controls weight given to symmetric Focal Tversky loss and symmetric Focal loss, by default 0.5
    delta : float, optional
        controls weight given to each class, by default 0.6
    gamma : float, optional
        focal parameter controls the degree of background suppression and foreground enhancement, by default 0.5
    """
    def loss_function(y_pred, y_true):

        ftl = focal_tversky_loss(delta=delta, gamma=gamma)(y_pred, y_true)
        fl = focal_loss(delta=delta, gamma=gamma)(y_pred, y_true)
        if weight is not None:
            return (weight * ftl) + ((1-weight) * fl)  
        else:
            return ftl + fl

    return loss_function


###########################################
#      Symmetric Unified Focal loss       #
###########################################
def sym_unified_focal_loss(weight=0.5, delta=0.6, gamma=0.5):
    """The Unified Focal loss is a new compound loss function that unifies Dice-based and cross entropy-based loss functions into a single framework.
    Parameters
    ----------
    weight : float, optional
        represents lambda parameter and controls weight given to symmetric Focal Tversky loss and symmetric Focal loss, by default 0.5
    delta : float, optional
        controls weight given to each class, by default 0.6
    gamma : float, optional
        focal parameter controls the degree of background suppression and foreground enhancement, by default 0.5
    """
    def loss_function(y_pred, y_true):

        focal_tversky_loss
        symmetric_ftl = focal_tversky_loss(delta=delta, gamma=gamma)(y_pred, y_true)
        symmetric_fl = symmetric_focal_loss(delta=delta, gamma=gamma)(y_pred, y_true)
        if weight is not None:
            return (weight * symmetric_ftl) + ((1-weight) * symmetric_fl)  
        else:
            return symmetric_ftl + symmetric_fl

    return loss_function

def coverage(input, target):
    assert input.shape[1] == target.shape[1] 
    assert input.shape[1] == 1
    intersection = (input*target).sum(dim=(-1,-2,-3)) 
    cardinality = target.sum(dim=(-1,-2,-3)) + 1e-10
    return (intersection/cardinality).mean()


def false_positive(input, target):
    assert input.shape[1] == target.shape[1] 
    assert input.shape[1] == 1
    intersection = (input*(1-target)).sum(dim=(-1,-2,-3)) 
    cardinality = (1-target).sum(dim=(-1,-2,-3))
    return (intersection/cardinality).mean()

def false_negative(input, target):
    assert input.shape[1] == target.shape[1] 
    assert input.shape[1] == 1
    intersection = ((1-input)*target).sum(dim=(-1,-2,-3)) 
    cardinality = (target).sum(dim=(-1,-2,-3))
    return (intersection/cardinality).mean()



def dice_score(input, target):
    '''
    Binary Dice score
    input - [batch_size,1,H,W,D], probability [0,1]
    target - binary mask [batch_size,1,H,W,D], 1 for foreground, 0 for background

    Dice = 2TP/(2TP + FP + FN) = 2X*Y/(|X| + |Y|)
    '''

    target = target.squeeze(1) # squeeze channel 
    input = input.squeeze(1) # squeeze channel
    epsilon = 1e-10

    intersection = 2*torch.sum(input * target, dim=(1,2,3)) + epsilon # [bs,]
    cardinality = torch.sum(input, dim=(1,2,3)) + torch.sum(target, dim=(1,2,3)) + epsilon # [bs,]
    dice_score = intersection / cardinality

    return dice_score.mean()


def dice_loss_custom(input, target):
    '''
    Binary Dice loss
    input - [batch_size,1,H,W,D], probability [0,1]
    target - binary mask [batch_size,1,H,W,D], 1 for foreground, 0 for background
    '''

    target = target.squeeze(1) # squeeze channel 
    input = input.squeeze(1) # squeeze channel
    
    intersection = 2*torch.sum(input * target, dim=(1,2,3)) + 1 # [bs,]
    cardinality = torch.sum(torch.pow(input,2) + torch.pow(target,2), dim=(1,2,3)) + 1 # [bs,]
    dice_score = intersection / cardinality

    return 1-dice_score.mean()


def dice_sfl(delta=0.7, gamma=2.):
    sfl = symmetric_focal_loss(delta, gamma)
    def loss_function(y_pred, y_true):
        return dice_loss_custom(y_pred, y_true) + sfl(y_pred, y_true)
    return loss_function