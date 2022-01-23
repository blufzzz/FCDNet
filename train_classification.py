from __future__ import print_function
from tensorboardX import SummaryWriter  
from IPython.core.debugger import set_trace
from datetime import datetime
import os
import shutil
import time
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score
from datasets import create_datasets
import yaml
from easydict import EasyDict as edict
import pandas as pd
import numpy as np
from collections import defaultdict
from matplotlib import pyplot as plt
from tqdm import tqdm
from joblib import Parallel, delayed
import torch
from torch import nn
from torch import autograd
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchio as tio

from utils import get_capacity, normalize_, save, DiceScoreBinary, DiceLossBinary, parse_args
from IPython.core.display import display, HTML

from multiprocessing import cpu_count
N_CPU = cpu_count()
SEED = 42


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
                augmentation=None, 
                is_train=True):


    phase_name = 'train' if is_train else 'val'
    loss_name = config.opt.criterion
    metric_dict = defaultdict(list)
    patch_size = config.dataset.patch_size
    patch_batch_size = config.dataset.patch_batch_size
    batch_size = config.opt.train_batch_size if is_train else config.opt.val_batch_size
    samples_per_volume = config.dataset.samples_per_volume
    queue_length = batch_size*samples_per_volume
    classification_threshold = config.model.classification_threshold
    sampler_type = config.dataset.sampler_type
    patch_fcd_threshold = config.dataset.patch_fcd_threshold 
    labels = dataloader.dataset.labels
    shuffle_train = config.dataset.shuffle_train if hasattr(config.dataset,'shuffle_train') else True
    top_k_list = config.dataset.top_k_list if hasattr(config.dataset, 'top_k_list') else [10, 50, 100]
    assert isinstance(top_k_list, list)

    # used to turn on/off gradients
    grad_context = torch.autograd.enable_grad if is_train else torch.no_grad
    with grad_context():
        iterator = enumerate(dataloader)

        # brain_tensor - [bs,C,H,W,D]
        # mask_tensor - [bs,1,H,W,D]
        # label_tensor - [bs,1,H,W,D]
        for iter_i, (brain_tensor, mask_tensor, label_tensor) in iterator:

            ###########################
            # SETUP PATCH DATALOADERS #
            ###########################
            if is_train:
                batch_size = len(brain_tensor)

                label = labels[iter_i] if shuffle_train and batch_size==1 else iter_i
                # from batch to list of subjects
                subjects_list = []
                for batch_index in range(batch_size):

                    n_fcd = label_tensor[batch_index].sum()
                    if sampler_type == 'weights': 
                        
                        n_brain = mask_tensor[batch_index].sum()
                        k = n_brain / n_fcd
                        sampling_map = mask_tensor[batch_index].clone()
                        sampling_map[label_tensor[batch_index].type(torch.bool)] = 2*int(k)
                        sampler = tio.data.WeightedSampler(patch_size, 'sampling_map')

                    elif sampler_type == 'labels':

                        sampling_map = mask_tensor[batch_index].clone()
                        sampling_map[label_tensor[batch_index].type(torch.bool)] = 2
                        label_sample_coeff = (mask_tensor[batch_index].sum() / label_tensor[batch_index].sum())
                        label_sample_coeff = label_sample_coeff.item()
                        sampler = tio.data.LabelSampler(
                                                        patch_size=patch_size,
                                                        label_name='sampling_map',
                                                        label_probabilities={0:0,
                                                        1:1,
                                                        2:1}, # label_sample_coeff
                                                        )
                    else:
                        raise RuntimeError('Unknown sampler_type!')

                    subject = tio.Subject(t1=tio.ScalarImage(tensor=brain_tensor[batch_index]),
                                          label=tio.LabelMap(tensor=label_tensor[batch_index]),
                                          sampling_map=sampling_map,
                                          n_fcd=n_fcd)

                    if augmentation is not None:
                        subject = augmentation(subject)
                    subjects_list.append(subject)

                subjects_dataset = tio.SubjectsDataset(subjects_list)
                
                patches_queue = tio.Queue(
                    subjects_dataset,
                    queue_length,
                    samples_per_volume,
                    sampler,
                    num_workers=4
                )
                patch_loader = DataLoader(
                    patches_queue,
                    batch_size=patch_batch_size,
                    num_workers=0,  # this must be 0
                )

            else:
                label = labels[iter_i]
                assert brain_tensor.shape[0] == 1
                n_fcd = label_tensor.sum()
                subject = tio.Subject(t1=tio.ScalarImage(tensor=brain_tensor[0]),
                                      label=tio.LabelMap(tensor=label_tensor[0]),
                                      n_fcd=n_fcd)

                patch_overlap = patch_size - (patch_size//4) # 0.75
                grid_sampler = tio.inference.GridSampler(
                    subject, # validation subject
                    patch_size,
                    patch_overlap
                )

                patch_loader = torch.utils.data.DataLoader(grid_sampler, batch_size=patch_batch_size)
                aggregator = tio.inference.GridAggregator(grid_sampler, overlap_mode='average')

            ########################
            # ITERATE OVER PATCHES #
            #############################################################################
            # number of FCD pixels in patch to be considered as FCD patch
            metric_dict_patch = defaultdict(list)
            targets_all = []
            probs_all = []
            preds_all = []

            for patch_i, patches_batch in enumerate(patch_loader):

                inputs = patches_batch['t1'][tio.DATA].to(device)  # [bs,C,p,p,p]
                targets = patches_batch['label'][tio.DATA].to(device) # [bs,1,p,p,p]
                n_fcd = patches_batch['n_fcd'].to(device).unsqueeze(1)
                
                targets_ = (targets.sum([-1,-2,-3]) / n_fcd) >= patch_fcd_threshold
                targets_ = targets_.type(torch.float32)

                print(targets_.flatten())

                logits = model(inputs)
                loss = criterion(logits, targets_) # [bs,1], [bs,1]
                
                if is_train:
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                else:
                    locations = patches_batch[tio.LOCATION]
                    # casting back to patch
                    outputs = torch.ones_like(targets)*logits[...,None,None,None].sigmoid() # [bs,1,p,p,p]
                    aggregator.add_batch(outputs, locations)

                #####################
                # per-PATCH METRICS #
                #####################
                # map to and remove last dim
                targets_np = targets_.squeeze(-1).detach().cpu().numpy().astype(int) # [bs,]
                prob_pred_np = torch.sigmoid(logits.squeeze(-1)).detach().cpu().numpy() # [bs,]
                targets_pred_np = (prob_pred_np >= classification_threshold).astype(int)
                
                targets_all += list(targets_np)
                probs_all += list(prob_pred_np)
                preds_all += list(targets_pred_np)

                metric_dict_patch[f'{loss_name}'].append(loss.item())
            ##############################################################################
            class_balance = np.mean(targets_all)
            metric_dict['class_balance'].append(class_balance)
            if class_balance == 0:
                print(f'Class_balance for {label} is 0!')

            for k,v in metric_dict_patch.items():
                metric_dict[k].append(np.mean(v))

            targets_all=np.array(targets_all)
            probs_all=np.array(probs_all)
            preds_all=np.array(preds_all)

            precision = precision_score(targets_all, preds_all, zero_division=0)
            recall = recall_score(targets_all, preds_all, zero_division=0)
            try:
                roc_auc = roc_auc_score(targets_all, probs_all)
            except Exception as e:
                print(e)
                roc_auc = 0

            metric_dict['precision'].append(precision)
            metric_dict['recall'].append(recall)
            metric_dict['roc_auc'].append(roc_auc)
            
            # Gris sample on validation
            if not is_train:

                ###########
                # HITRATE #
                ###########
                # sorting by predicted probabilities
                argsort = np.argsort(probs_all, axis=0)[::-1]
                for top_k in top_k_list:
                    top_k_fcd = targets_all[argsort][:top_k]
                    hitrate = top_k_fcd.mean() / top_k #((1./(np.arange(top_k)+1))*top_k_fcd).sum() 
                    metric_dict[f'top-{top_k}_hitrate'].append(hitrate)

                ########
                # DICE #
                ########
                output_tensor = aggregator.get_output_tensor().unsqueeze(1) # [1,1,H,W,D]
                output_tensor = output_tensor * mask_tensor # zeros all non mask values
                dice = DiceScoreBinary(output_tensor, label_tensor).item()
                coverage = (output_tensor*label_tensor).sum() / label_tensor.sum()
                metric_dict['dice_score'].append(dice)
                metric_dict['coverage'].append(coverage.item())

            #########
            # PRINT #
            #########
            message = f'For {phase_name}, {label},'
            for title, value in metric_dict.items():
                v = np.round(value[-1],3)
                message+=f' {title}:{v}'
            print(message)

            if writer is not None:
                for title, value in metric_dict.items():
                    writer.add_scalar(f"{phase_name}_{title}", value[-1], n_iters_total)

            n_iters_total += 1
    
    ###################
    # PER-EPOCH STATS #
    ###################
    for title, value in metric_dict.items():
        m = np.mean(value)
        metric_dict_epoch[phase_name + '_' + title].append(m)
        if writer is not None:
            writer.add_scalar(f"{phase_name}_{title}_epoch", m, epoch)

    return n_iters_total, target_metric

def main(args):


    print(f'Available devices: {torch.cuda.device_count()}')

    with open(args.config) as fin:
        config = edict(yaml.safe_load(fin))

    # setting logs
    MAKE_LOGS = config.make_logs
    SAVE_MODEL = config.opt.save_model if hasattr(config.opt, "save_model") else True
    DEVICE = config.opt.device if hasattr(config.opt, "device") else 1
    device = torch.device(DEVICE)
    assert config.opt.val_batch_size == 1

    experiment_name = '{}@{}'.format(args.experiment_comment, datetime.now().strftime("%d.%m.%Y-%H:%M:%S"))
    print("Experiment name: {}".format(experiment_name))
    
    if MAKE_LOGS:
        experiment_dir = os.path.join(args.logdir, experiment_name)
        if os.path.isdir(experiment_dir):
            shutil.rmtree(experiment_dir)
        os.makedirs(experiment_dir)
        shutil.copy(args.config, os.path.join(experiment_dir, "config.yaml"))
    writer = SummaryWriter(os.path.join(experiment_dir, "tb")) if MAKE_LOGS else None

    ################
    # CREATE MODEL #
    ################
    model = torchvision.models.video.r3d_18(pretrained=False, progress=False)
    conv3d_1 = model.stem[0]
    # features
    if config.dataset.features == 'ALL':
        input_channels = 10
    else:
        assert isinstance(config.dataset.features, list)
        input_channels = len(config.dataset.features)

    model.stem[0] = nn.Conv3d(in_channels=input_channels,
                             out_channels=conv3d_1.out_channels,
                             kernel_size=conv3d_1.kernel_size,
                             padding=conv3d_1.padding,
                             bias=conv3d_1.bias)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model = model.to(device)
    model_capacity = get_capacity(model)
    print(f'Model created! Capacity: {model_capacity}')

    if hasattr(config.model, 'weights'):
        model_dict = torch.load(os.path.join(config.model.weights, 'checkpoints/weights.pth'))
        print(f'LOADING from {config.model.weights} \n epoch:', model_dict['epoch'])
        model.load_state_dict(model_dict['model_state'])

    ###################
    # CREATE DATASETS #
    ###################
    train_dataset, val_dataset = create_datasets(config)
    collate_fn = None
    shuffle_train = config.dataset.shuffle_train if hasattr(config.dataset,'shuffle_train') else True
    train_dataloader = DataLoader(train_dataset,
                                    batch_size=config.opt.train_batch_size,
                                    shuffle=shuffle_train,
                                    collate_fn=collate_fn)

    val_dataloader = DataLoader(val_dataset,
                                batch_size=config.opt.val_batch_size,
                                shuffle=False,
                                collate_fn=collate_fn)

    print(len(train_dataloader), len(val_dataloader))

    augmentation = None
    if config.opt.augmentation:
        symmetry = tio.RandomFlip(axes=0) 
        bias = tio.RandomBiasField(coefficients=0.3)
        noise = tio.RandomNoise(std=(0,1e-3))
        affine = tio.RandomAffine(scales=(0.9, 1.1, 0.9, 1.1, 0.9, 1.1), 
                                 degrees=5,
                                 translation=(1,1,1),
                                 center='image',
                                 default_pad_value=0)
        rescale = tio.RescaleIntensity(out_min_max=(0, 1))
        augmentation = tio.Compose([symmetry, bias, noise, affine, rescale])

    ################
    # CREATE OPTIM #
    ################
    criterion = {
        "BCE": nn.BCEWithLogitsLoss(), # [logits:float32, target:float32]
    }[config.opt.criterion]
    opt = optim.Adam(model.parameters(), lr=config.opt.lr)


    ############
    # TRAINING #
    ############
    print('Start training!')
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
                                            train_dataloader, 
                                            device, 
                                            writer, 
                                            epoch, 
                                            metric_dict_epoch, 
                                            n_iters_total_train,
                                            augmentation=augmentation,
                                            is_train=True)
            print (f'VAL EPOCH: {epoch} ... ')
            n_iters_total_val, target_metric = one_epoch(model, 
                                                criterion, 
                                                opt, 
                                                config, 
                                                val_dataloader, 
                                                device, 
                                                writer, 
                                                epoch, 
                                                metric_dict_epoch, 
                                                n_iters_total_val,
                                                augmentation=None,
                                                is_train=False)

            if SAVE_MODEL and MAKE_LOGS:
                if target_metric > target_metric_prev:
                    print(f'target_metric = {target_metric}, SAVING...')
                    save(experiment_dir, model, opt, epoch)
                    target_metric_prev = target_metric

    except Exception as e:
        set_trace()
        # keyboard interrupt
        if MAKE_LOGS:
            np.save(os.path.join(experiment_dir, 'metric_dict_epoch'), metric_dict_epoch)

if __name__ == '__main__':
    args = parse_args()
    main(args)
    
