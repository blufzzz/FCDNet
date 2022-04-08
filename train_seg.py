from tensorboardX import SummaryWriter  
from IPython.core.debugger import set_trace
import traceback
from datetime import datetime
import os
import shutil
import time
from collections import defaultdict
import pickle
import numpy as np
import torch
from torch import nn
from torch import autograd
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from utils import save, parse_args, get_capacity
from models.v2v import V2VModel
import torchio as tio
from datasets import create_datasets
import yaml
from easydict import EasyDict as edict
from losses import focal_tversky_loss, DiceScoreBinary, DiceLossBinary


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
    target_metric_name = config.model.target_metric_name 

    # used to turn on/off gradients
    grad_context = torch.autograd.enable_grad if is_train else torch.no_grad
    with grad_context():
        iterator = enumerate(dataloader)
        val_predictions = {}
        for iter_i, (brain_tensor, mask_tensor, label_tensor) in iterator:
            

            if (augmentation is not None) and is_train:
                subject = tio.Subject(
                    t1=tio.ScalarImage(tensor=brain_tensor[0]),
                    label=tio.LabelMap(tensor=label_tensor[0]),
                    mask=tio.LabelMap(tensor=mask_tensor[0]),
                    diagnosis='positive'
                )

                transformed = augmentation(subject)
                brain_tensor = transformed['t1'].tensor.unsqueeze(0)
                label_tensor = transformed['label'].tensor.unsqueeze(0)
                mask_tensor = transformed['mask'].tensor.unsqueeze(0)

            if config.interpolate:
                brain_tensor = F.interpolate(brain_tensor, config.interpolation_size).to(device)
                label_tensor = F.interpolate(label_tensor, config.interpolation_size).to(device)
                mask_tensor = F.interpolate(mask_tensor, config.interpolation_size).to(device)
            else:
                brain_tensor = brain_tensor.to(device)
                label_tensor = label_tensor.to(device)
                mask_tensor = mask_tensor.to(device)


            # forward pass
            label_tensor_predicted = model(brain_tensor) # -> [bs,1,ps,ps,ps]
            
            if config.opt.criterion == "BCE":
                weights = torch.ones(label_tensor.shape).to(device)
                weights[label_tensor > 0] = config.opt.bce_weights   #250
                loss = criterion(weight=weights, reduction='mean')(label_tensor_predicted, label_tensor)
            elif config.opt.criterion == "DiceBCE":
                weights = torch.ones(label_tensor.shape).to(device)
                weights[label_tensor > 0] = config.opt.bce_weights   #250
                loss = torch.nn.BCELoss(weight=weights, reduction='mean')(label_tensor_predicted, label_tensor) + \
                        DiceLossBinary(label_tensor_predicted, label_tensor)
            else:
                loss = criterion(label_tensor_predicted, label_tensor) 


            if is_train:
                opt.zero_grad()
                loss.backward()
                opt.step()
            
            metric_dict[f'{loss_name}'].append(loss.item())
            label_tensor_predicted = label_tensor_predicted*mask_tensor
            dice_score = DiceScoreBinary(label_tensor_predicted, label_tensor)
            coverage = (label_tensor_predicted*label_tensor).sum() / label_tensor.sum()
            
            if not is_train:
                label = dataloader.dataset.labels[iter_i]
                val_predictions[label] = label_tensor_predicted.detach().cpu().numpy()
            
            metric_dict['coverage'].append(coverage.item())
            metric_dict['dice_score'].append(dice_score.item())

            print(f'Epoch: {epoch}, iter: {iter_i}, Loss_{loss_name}: {loss.item()}, DICE: {dice_score.item()}')

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
                        torch.save(pred,
                                    os.path.join(config.dataset.val_preds_path, f'{label}'))


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

    experiment_name = '{}@{}'.format(args.experiment_comment, datetime.now().strftime("%d.%m.%Y-%H:%M:%S"))
    print("Experiment name: {}".format(experiment_name))

    
    if MAKE_LOGS:
        experiment_dir = os.path.join(args.logdir, experiment_name)
        if os.path.isdir(experiment_dir):
            shutil.rmtree(experiment_dir)
        os.makedirs(experiment_dir)
        shutil.copy(args.config, os.path.join(experiment_dir, "config.yaml"))
        if config.dataset.save_best_val_predictions:
            val_preds_path = os.path.join(experiment_dir, 'best_val_preds')
            config.dataset.val_preds_path = val_preds_path
            os.makedirs(val_preds_path)

    writer = SummaryWriter(os.path.join(experiment_dir, "tb")) if MAKE_LOGS else None
    model = V2VModel(config).to(device)
    capacity = get_capacity(model)
    print(f'Model created! Capacity: {capacity}')

    if hasattr(config.model, 'weights'):
        model_dict = torch.load(os.path.join(config.model.weights, 'checkpoints/weights.pth'))
        print(f'LOADING from {config.model.weights} \n epoch:', model_dict['epoch'])
        model.load_state_dict(model_dict['model_state'])

    ##################
    # CREATE DATASETS #
    ###################
    train_dataset, val_dataset = create_datasets(config)
    train_dataloader = DataLoader(train_dataset, batch_size=config.opt.train_batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config.opt.val_batch_size, shuffle=False)
    # for the proper torchio augmentation
    assert config.opt.train_batch_size == 1
    assert config.opt.val_batch_size == 1
    print(len(train_dataloader), len(val_dataloader))


    transform = None
    if config.opt.augmentation:
        symmetry = tio.RandomFlip(axes=0) 
        noise = tio.RandomNoise(std=(0,1e-3))
        blur = tio.RandomBlur((0,1e-1))
        affine = tio.RandomAffine(scales=(0.95, 1.05, 0.95, 1.05, 0.95, 1.05), 
                                 degrees=3,
                                 translation=(1,1,1),
                                 center='image',
                                 default_pad_value=0)
        rescale = tio.RescaleIntensity(out_min_max=(0, 1))
        transform = tio.Compose([symmetry, blur, noise, affine, rescale])


    ################
    # CREATE OPTIM #
    ################
    criterion = {
        "BCE":torch.nn.BCELoss, # [probabilities, target]
        "Dice":DiceLossBinary,
        "DiceBCE":None,
        "FocalTversky":focal_tversky_loss(),
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
                                            augmentation=transform,
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
        set_trace()
        # keyboard interrupt
        if MAKE_LOGS:
            np.save(os.path.join(experiment_dir, 'metric_dict_epoch'), metric_dict_epoch)

if __name__ == '__main__':
    args = parse_args()
    print("args: {}".format(args))
    main(args)
