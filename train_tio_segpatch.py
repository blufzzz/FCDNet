from tensorboardX import SummaryWriter  
from IPython.core.debugger import set_trace
import traceback
from datetime import datetime
import os
import shutil
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
from datasets import create_datasets
import yaml
from easydict import EasyDict as edict
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import torch
from torch import nn
from torch import autograd
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchio as tio
from utils import get_capacity, save, DiceScoreBinary, DiceLossBinary, parse_args
from models.v2v import V2VModel

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
    labels = dataloader.dataset.labels
    patch_overlap = config.dataset.patch_overlap
    assert batch_size == 1
    val_predictions = {}
    target_metric_name = config.model.target_metric_name if hasattr(config.model,'target_metric_name') else 'dice'

    pov = int(patch_size*patch_overlap) # take high overlap to avoid missing
    if pov%2!=0:
        pov+=1

    # used to turn on/off gradients
    grad_context = torch.autograd.enable_grad if is_train else torch.no_grad
    with grad_context():

        # bs = 1
        # brain_tensor - [1,C,H,W,D]
        # mask_tensor - [1,1,H,W,D]
        # label_tensor - [1,1,H,W,D]
        #######################
        # ITERATE OVER BRAINS #
        #######################
        for iter_i, (brain_tensor, mask_tensor, label_tensor) in enumerate(dataloader):

            ###########################
            # SETUP PATCH DATALOADERS #
            ###########################
            label = labels[iter_i]

            if label_tensor.sum() == 0:
                print(f'No label for {label}, skipping...')
                continue
            else:
                print(f'Label: {label}', label_tensor.sum())

            subject = tio.Subject(t1=tio.ScalarImage(tensor=brain_tensor[0]),label=tio.LabelMap(tensor=label_tensor[0]))
            
            if is_train and (augmentation is not None):
                subject = augmentation(subject)
                # make 0 background
            
            grid_sampler = tio.inference.GridSampler(subject, patch_size, pov)
            patch_loader = torch.utils.data.DataLoader(grid_sampler, batch_size=patch_batch_size, shuffle=True)
            aggregator = tio.inference.GridAggregator(grid_sampler, overlap_mode='average')

            ########################
            # ITERATE OVER PATCHES #
            #############################################################################
            # number of FCD pixels in patch to be considered as target patch
            metric_dict_patch = defaultdict(list)
            print(f'Iterating for {label}, {len(patch_loader)}')
            for patch_i, patches_batch in enumerate(patch_loader):

                inputs = patches_batch['t1'][tio.DATA].to(device)  # [bs,C,p,p,p]
                targets = patches_batch['label'][tio.DATA].to(device) # [bs,1,p,p,p]

                logits = model(inputs)

                if config.opt.criterion == "BCE":
                    weights = torch.ones(targets.shape).to(device)
                    weights[targets > 0] = config.opt.bce_weights   #250
                    loss = criterion(weight=weights, reduction='mean')(logits, targets)
                elif config.opt.criterion == "DiceBCE":
                    weights = torch.ones(targets.shape).to(device)
                    weights[targets > 0] = config.opt.bce_weights   #250
                    loss = torch.nn.BCELoss(weight=weights, reduction='mean')(logits, targets) + \
                            DiceLossBinary(logits, targets)
                else:
                    loss = criterion(logits, targets) # [bs,1,p,p,p], [bs,1,p,p,p]


                
                if is_train:
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

                locations = patches_batch[tio.LOCATION]

                # casting back to patch
                aggregator.add_batch(logits.detach(), locations)

                #####################
                # per-PATCH METRICS #
                #####################
                metric_dict_patch[f'{loss_name}'].append(loss.item())
            
            ##############################################################################
            for k,v in metric_dict_patch.items():
                metric_dict[k].append(np.mean(v))

            ########
            # DICE #
            ########
            output_tensor = aggregator.get_output_tensor().unsqueeze(1) # [1,1,H,W,D]
            output_tensor = output_tensor * mask_tensor # zeros all non mask values
            dice = DiceScoreBinary(output_tensor, label_tensor).item()
            coverage = (output_tensor*label_tensor).sum() / label_tensor.sum()
            metric_dict['dice_score'].append(dice)
            metric_dict['coverage'].append(coverage.item())
            if not is_train:
                val_predictions[label] = output_tensor.detach().cpu().numpy()

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
    target_metric = 0
    for title, value in metric_dict.items():
        m = np.mean(value)
        metric_dict_epoch[phase_name + '_' + title].append(m)
        if writer is not None:
            writer.add_scalar(f"{phase_name}_{title}_epoch", m, epoch)
        if title == config.model.target_metric_name:
            target_metric = m

    #####################
    # SAVING BEST PREDS #
    #####################
    target_metrics_epoch = metric_dict_epoch[f'val_{target_metric_name}']
    if not is_train:
        if config.dataset.save_best_val_predictions:
            if len(target_metrics_epoch) == 1 or target_metrics_epoch[-1] >= target_metrics_epoch[-2]:
                for label,pred in val_predictions.items():
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
    # essential for the proper samplers functioning
    assert config.opt.val_batch_size == 1
    assert config.opt.train_batch_size == 1
    assert config.dataset.sampler_type == 'grid'
    # assert config.dataset.shuffle_train == False

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

    ################
    # CREATE MODEL #
    ################
    model = V2VModel(config).to(device)
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
    
    train_dataloader = DataLoader(train_dataset,
                                    batch_size=config.opt.train_batch_size,
                                    shuffle=config.dataset.shuffle_train,
                                    collate_fn=collate_fn)

    val_dataloader = DataLoader(val_dataset,
                                batch_size=config.opt.val_batch_size,
                                shuffle=False,
                                collate_fn=collate_fn)

    print(len(train_dataloader), len(val_dataloader))

    augmentation = None
    if config.opt.augmentation:
        symmetry = tio.RandomFlip(axes=0) 
        bias = tio.RandomBiasField(coefficients=0.1)
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
        "Dice": DiceLossBinary, 
        "BCE": nn.BCELoss,
        "DiceBCE":None
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
                if not config.model.use_greedy_saving:
                    print(f'SAVING...')
                    save(experiment_dir, model, opt, epoch)
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
    main(args)
    



# n_iters_total_train, _  = one_epoch(model, criterion, opt, config, train_dataloader, device, writer, epoch, metric_dict_epoch, n_iters_total_train, augmentation=augmentation, is_train=True)