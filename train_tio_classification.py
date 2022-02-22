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
from utils import get_capacity, save, DiceScoreBinary, parse_args

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
    classification_threshold = config.model.classification_threshold
    patch_fcd_threshold = config.dataset.patch_fcd_threshold 
    labels = dataloader.dataset.labels
    patch_overlap = config.dataset.patch_overlap
    top_k_list = config.dataset.top_k_list if hasattr(config.dataset, 'top_k_list') else [10, 50, 100]
    assert isinstance(top_k_list, list)
    assert batch_size == 1

    pov = int(patch_size*patch_overlap) # take high overlap to avoid missing
    if pov%2!=0:
        pov+=1

    # used to turn on/off gradients
    grad_context = torch.autograd.enable_grad if is_train else torch.no_grad
    with grad_context():
        iterator = enumerate(dataloader)

        # bs = 1
        # brain_tensor - [1,C,H,W,D]
        # mask_tensor - [1,1,H,W,D]
        # label_tensor - [1,1,H,W,D]
        #######################
        # ITERATE OVER BRAINS #
        #######################
        for iter_i, (brain_tensor, mask_tensor, label_tensor) in iterator:

            ###########################
            # SETUP PATCH DATALOADERS #
            ###########################
            label = labels[iter_i]

            if label_tensor.sum() == 0:
                print(f'No label for {label}, skipping...')
                continue

            n_fcd = label_tensor.sum()
            subject = tio.Subject(t1=tio.ScalarImage(tensor=brain_tensor[0]),
                          label=tio.LabelMap(tensor=label_tensor[0]),
                          n_fcd=n_fcd)
            
            if is_train and (augmentation is not None):
                subject = augmentation(subject)

            

            grid_sampler = tio.inference.GridSampler(subject, patch_size, pov)
            patch_loader = torch.utils.data.DataLoader(grid_sampler, batch_size=patch_batch_size)
            aggregator = tio.inference.GridAggregator(grid_sampler, overlap_mode='average')

            ########################
            # ITERATE OVER PATCHES #
            #############################################################################
            # number of FCD pixels in patch to be considered as target patch
            metric_dict_patch = defaultdict(list)
            targets_all = []
            probs_all = []
            preds_all = []
            print(f'Iterating for {label}, {len(patch_loader)}')
            for patch_i, patches_batch in enumerate(patch_loader):

                inputs = patches_batch['t1'][tio.DATA].to(device)  # [bs,C,p,p,p]
                targets = patches_batch['label'][tio.DATA].to(device) # [bs,1,p,p,p]
                n_fcd = patches_batch['n_fcd'].to(device).unsqueeze(1)
                
                # it is target crop if there is enough pixels percentage of the target
                targets_ = (targets.sum([-1,-2,-3]) / n_fcd) >= patch_fcd_threshold
                # it is target crop if there is enough pixels in patch
                targets_ += (targets.sum([-1,-2,-3]) / (patch_size**3)) >= patch_fcd_threshold
                targets_ = targets_.type(torch.float32)

                logits = model(inputs)
                loss = criterion(logits, targets_) # [bs,1], [bs,1]
                
                if is_train:
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

                locations = patches_batch[tio.LOCATION]

                # casting back to patch
                outputs = torch.ones_like(targets)*logits[...,None,None,None].sigmoid() # [bs,1,p,p,p]
                aggregator.add_batch(outputs.detach(), locations)

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
            print(f'Class_balance for {label} is {class_balance}!')
            if class_balance == 0:
                print('Skipping!')
                continue

            for k,v in metric_dict_patch.items():
                metric_dict[k].append(np.mean(v))

            targets_all=np.array(targets_all)
            probs_all=np.array(probs_all)
            preds_all=np.array(preds_all)

            accuracy = accuracy_score(targets_all, preds_all)
            precision = precision_score(targets_all, preds_all, zero_division=0)
            recall = recall_score(targets_all, preds_all, zero_division=0)
            roc_auc = roc_auc_score(targets_all, probs_all, average='samples')
            f1 = f1_score(targets_all, preds_all, average='weighted')

            metric_dict['accuracy'].append(accuracy)
            metric_dict['precision'].append(precision)
            metric_dict['recall'].append(recall)
            metric_dict['roc_auc_samples'].append(roc_auc)
            metric_dict['f1_weighted'].append(f1)

            ###########
            # HITRATE #
            ###########
            # sorting by predicted probabilities
            argsort = np.argsort(probs_all, axis=0)[::-1]
            for top_k in top_k_list:
                top_k_fcd = targets_all[argsort][:top_k]
                hitrate = top_k_fcd.mean()
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
    target_metric = 0
    for title, value in metric_dict.items():
        m = np.mean(value)
        metric_dict_epoch[phase_name + '_' + title].append(m)
        if writer is not None:
            writer.add_scalar(f"{phase_name}_{title}_epoch", m, epoch)
        if title == config.opt.target_metric_name:
            target_metric = m
        else:
            print('No target metric was found!')

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
    assert config.dataset.shuffle_train == False

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
    
