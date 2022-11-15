from datetime import datetime
import re, time, os, shutil, json
import numpy as np
import tempfile
import configdot
import traceback

from collections import defaultdict
from tensorboardX import SummaryWriter  
from IPython.core.debugger import set_trace

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
import torch.optim as optim
from models.v2v import V2VModel
from torchinfo import summary

from losses import *
from dataset import setup_dataloaders, setup_datafiles, create_datafile, setup_transformations
from utils import save, get_capacity, calc_gradient_norm, get_label, to_numpy, show_prediction_slice
import monai
from monai.config import print_config
from monai.utils import set_determinism
import matplotlib.pyplot as plt

from monai.data import create_test_image_3d, list_data_collate, decollate_batch, pad_list_data_collate, DataLoader, Dataset, CacheDataset
from monai.apps import CrossValidation
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
import multiprocessing
from multiprocessing import Pool
import time

plt.ion() 

print_config()


# enable cuDNN benchmark
# torch.backends.cudnn.benchmark = True
# torch.manual_seed(42)
# torch.use_deterministic_algorithms(True)

def one_epoch(model, 
              criterion, 
              opt,
              config, 
              dataloader, 
              device, 
              writer, 
              epoch, 
              metric_dict_epoch, 
              n_iters_total=0,
              is_train=True):

    # use amp to accelerate training
    if config.opt.use_scaler:
        scaler = torch.cuda.amp.GradScaler()

    phase_name = 'train' if is_train else 'val'
    loss_name = config.opt.criterion
    metric_dict = defaultdict(list)
    target_metric_name = config.model.target_metric_name 

    if not is_train:
        model.eval()
    else:
        model.train()

    # used to turn on/off gradients
    grad_context = torch.autograd.enable_grad if is_train else torch.no_grad
    with grad_context():
        iterator = enumerate(dataloader)
        val_predictions = {}
        for iter_i, data_tensors in iterator:
            brain_tensor, label_tensor, mask_tensor = (data_tensors['image'].to(device),
                                                       data_tensors['seg'].to(device),
                                                       data_tensors['mask'].to(device)
                                                       )
            
            # forward pass
            t1 = time.time()
            with autocast(enabled=config.opt.use_scaler):
                label_tensor_predicted = model(brain_tensor) # -> [bs,1,ps,ps,ps]
                loss = criterion(label_tensor_predicted, label_tensor) 

            if is_train:
                opt.zero_grad()

                if config.opt.use_scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                if hasattr(config.opt, "grad_clip"):
                    if config.opt.use_scaler:
                        scaler.unscale_(opt)
                        torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                           config.opt.grad_clip)

                metric_dict['grad_norm'].append(calc_gradient_norm(filter(lambda x: x[1].requires_grad, 
                                                model.named_parameters())))

                if config.opt.use_scaler:
                    scaler.step(opt)
                    scaler.update()
                else:
                    opt.step()

            t2 = time.time()    
            dt = t2-t1 # inference time
            
            metric_dict[f'batch_time'].append(dt)
            metric_dict[f'{loss_name}'].append(loss.item())
            label_tensor_predicted = label_tensor_predicted * mask_tensor
            
            if hasattr(config.opt, 'save_inference') and config.opt.save_inference:
                show_prediction_slice(iter_i, 
                                      brain_tensor, 
                                      mask_tensor, 
                                      label_tensor, 
                                      label_tensor_predicted, 
                                      b_ind=0, 
                                      c_ind=0)
            
            cov = coverage(label_tensor_predicted, label_tensor).item()
            fp = false_positive(label_tensor_predicted, label_tensor).item()
            fn = false_negative(label_tensor_predicted, label_tensor).item()
            dice = dice_score(label_tensor_predicted, label_tensor).item()
            
            if not is_train and config.dataset.save_best_val_predictions:
                label = get_label(dataloader.dataset.data[iter_i]['seg'])
                val_predictions[label] = label_tensor_predicted.detach().cpu().numpy()
            
            metric_dict['coverage'].append(cov) # a.k.a recall
            metric_dict['false_positive'].append(fp)
            metric_dict['false_negative'].append(fn)
            metric_dict['dice_score'].append(dice)
            
            #########
            # PRINT #
            #########
            message = f'For {phase_name}, iter: {iter_i},'
            for title, value in metric_dict.items():
                if title == 'grad_norm':
                    v = np.round(value[-1],6)
                else:
                    v = np.round(value[-1],3)
                message+=f' {title}:{v}'
            print(message)


            if is_train and writer is not None:
                for title, value in metric_dict.items():
                    writer.add_scalar(f"{phase_name}_{title}", value[-1], n_iters_total)

            n_iters_total += 1

    target_metric = 0
    for title, value in metric_dict.items():
        m = np.mean(value)
        metric_dict_epoch[phase_name + '_' + title].append(m)
        if title == target_metric_name:
            target_metric = m
        if writer is not None:
            writer.add_scalar(f"{phase_name}_{title}_epoch", m, epoch)
            
    #####################
    # SAVING BEST PREDS #
    #####################
    target_metrics_epoch = metric_dict_epoch[f'val_{target_metric_name}']
    if not is_train:
        if config.dataset.save_best_val_predictions:
            # use greedy-saving: save only if the target metric is improved
            if len(target_metrics_epoch) == 1 or target_metrics_epoch[-1] >= target_metrics_epoch[-2]:
                for label, pred in val_predictions.items():
                    torch.save(pred, os.path.join(config.dataset.val_preds_path, f'{label}'))

    return n_iters_total, target_metric


def main(i):
    
    config = configdot.parse_config('configs/config-cv.ini')
    ##################
    # SETTING DEVICE #
    ##################
    print(torch.cuda.is_available())
    DEVICE = config.opt.device if hasattr(config.opt, "device") else 1
    device = torch.device(DEVICE)
    torch.cuda.set_device(DEVICE)

    print('Setting GPU#:', DEVICE)
    print('Using GPU#:', torch.cuda.current_device())

    BASE_DIR = '/workspace/RawData/Features'
    OUTPUT_DIR = '/workspace/RawData/Features/BIDS'
    TMP_DIR = '/workspace/Features/tmp'
    
    ##########
    # LOGDIR #
    ##########
    MAKE_LOGS = config.default.make_logs
    SAVE_MODEL = config.opt.save_model if hasattr(config.opt, "save_model") else True

    experiment_name = '{}@{}@fold-{}'.format(config.default.experiment_comment, datetime.now().strftime("%d.%m.%Y-%H"), i)
    print("Experiment name: {}".format(experiment_name))

    writer = None
    if MAKE_LOGS:
        # create experiment dir
        experiment_dir = os.path.join(config.default.log_dir, experiment_name)
        if os.path.isdir(experiment_dir):
            shutil.rmtree(experiment_dir)
        os.makedirs(experiment_dir)
        shutil.copy('configs/config-cv.ini', os.path.join(experiment_dir, "config-cv.ini"))

        # create dir for best_val_predictions
        if config.dataset.save_best_val_predictions:
            val_preds_path = os.path.join(experiment_dir, 'best_val_preds')
            config.dataset.val_preds_path = val_preds_path
            os.makedirs(val_preds_path)

        writer = SummaryWriter(os.path.join(experiment_dir, "tb"))

    #########
    # MODEL #
    #########
    if config.model.name == "v2v":
        model = V2VModel(config).to(device)
    else:
        model = VNet(spatial_dims=3, in_channels=2, out_channels=1,dropout_prob=0.1, dropout_dim=3).to(device)
        
    summary(model)

    capacity = get_capacity(model)
    print(f'Model created! Capacity: {capacity}')

    if hasattr(config.model, 'weights'):
        model_dict = torch.load(config.model.weights, map_location='cpu')
        print(f'LOADING from {config.model.weights} \n epoch:', model_dict['epoch'])
        model.load_state_dict(model_dict['model_state'])#.to(device)

    ################
    # CREATE OPTIM #
    ################
    criterion = {
        "BCE":bce_weighted(delta=config.opt.delta), # [probabilities, target]
        "Dice":dice_loss_custom,
        "DiceSFL": dice_sfl(delta=config.opt.delta, gamma=config.opt.gamma),
        "TL": tversky_loss(delta=config.opt.delta),
        "FTL": focal_tversky_loss(delta=config.opt.delta, gamma=config.opt.gamma),
        "SFL": symmetric_focal_loss(delta=config.opt.delta, gamma=config.opt.gamma),
        "USFL": sym_unified_focal_loss(weight=config.opt.weight,
                                         delta=config.opt.delta,
                                         gamma=config.opt.gamma)
    }[config.opt.criterion]


    opt = optim.Adam(model.parameters(), lr=config.opt.lr)
    
    
    ######################
    # CREATE DATALOADERS #
    ######################
    
    metadata_path = config.dataset.metadata_path
    
    scaling_dict = None
    if config.dataset.scaling_method in 'torchio':
        scaling_dict = 'torchio'
    elif config.dataset.scaling_method in 'scale_metadata':
        
        scaling_data_path = config.dataset.scaling_metadata_path
        scaling_dict = np.load(scaling_data_path, allow_pickle=True).item()
    else:
        print('Warning! no SCALING METADATA used! Applying naive independent MinMax...')
    
    split_dict = np.load(metadata_path, allow_pickle=True)
    train_list = split_dict[i].get('train')
    val_list = split_dict[i].get('val')

    images_list = []
    feat_params = config.dataset.features
    
    # Flag to add mask as additional sequence to Subset
    add_mask = config.dataset.trim_background
    
    train_files, train_missing_files = create_datafile(train_list, feat_params, mask=add_mask)
    val_files, val_missing_files = create_datafile(val_list, feat_params, mask=add_mask)
    
    print(scaling_dict)
    train_transf, val_transf = setup_transformations(config, scaling_dict)
    
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

    print(f'Start training for fold {i}')
    print(split_dict[i])
    np.save(os.path.join(experiment_dir, f'dataset-fold-{i}.npy'), split_dict[i])
    
    metric_dict_epoch = defaultdict(list)
    n_iters_total_train = 0 
    n_iters_total_val = 0
    target_metric = 0
    target_metric_prev = -1
    try:
        for epoch in range(config.opt.start_epoch, config.opt.n_epochs):
            
            print (f'TRAIN EPOCH: {epoch} ... ')
            n_iters_total_train, _  = one_epoch(model, 
                                            criterion, 
                                            opt, 
                                            config, 
                                            train_loader, 
                                            device, 
                                            writer, 
                                            epoch, 
                                            metric_dict_epoch, 
                                            n_iters_total_train,
                                            is_train=True)
            
            print (f'VAL EPOCH: {epoch} ... ')
            n_iters_total_val, target_metric = one_epoch(model, 
                                            criterion, 
                                            opt, 
                                            config, 
                                            val_loader, 
                                            device, 
                                            writer, 
                                            epoch, 
                                            metric_dict_epoch, 
                                            n_iters_total_val,
                                            is_train=False)

            if SAVE_MODEL and MAKE_LOGS:
                if not config.model.use_greedy_saving:
                    print(f'SAVING...')
                    save(experiment_dir, model, opt, epoch)
                # use greedy-saving: save only if the target metric is improved
                elif target_metric > target_metric_prev:
                    print(f'target_metric = {target_metric}, SAVING...')
                    save(experiment_dir, model, opt, epoch)
                    target_metric_prev = target_metric
                    
    except Exception as e:
        print(traceback.format_exc())
        #set_trace()

        
if __name__ == '__main__':
    set_determinism(seed=42)
    os.makedirs('./MONAI_TMP', exist_ok=True)
    os.environ['MONAI_DATA_DIRECTORY'] = "./MONAI_TMP"
    directory = os.environ.get("MONAI_DATA_DIRECTORY")
    root_dir = tempfile.mkdtemp() if directory is None else directory
    #args = None # everything in `config.ini` now
    conf = configdot.parse_config('configs/config-cv.ini')
    s = conf.dataset.ind_fold_start
    e = conf.dataset.ind_fold_last
    #num = 9  # Number of folds
    p = multiprocessing.Pool(processes = 2)
    start = time.time()
    for i in range(s,e):
        p.apply_async(main, [i])
    p.close()
    p.join()
    print('Complete')
    end = time.time()
    print('total time (s)= ' + str(end-start))
