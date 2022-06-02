from tensorboardX import SummaryWriter  
from IPython.core.debugger import set_trace
import traceback
from datetime import datetime
import os
import shutil
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
from datasets import create_datasets, BrainPatchesDataset, BalancedSampler, add_xyz
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
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchio as tio
from losses import DiceScoreBinary, DiceLossBinary, sym_unified_focal_loss, compute_BCE, symmetric_focal_loss
from utils import get_capacity, save, parse_args, calc_gradient_norm
from models.v2v import V2VModel

# enable cuDNN benchmark
torch.backends.cudnn.benchmark = True
# torch.manual_seed(42)


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
    
    # use amp to accelerate training
    if config.opt.use_scaler:
        scaler = torch.cuda.amp.GradScaler()

    phase_name = 'train' if is_train else 'val'
    loss_name = config.opt.criterion
    metric_dict = defaultdict(list)
    patch_size = config.dataset.patch_size
    patch_batch_size = config.dataset.patch_batch_size
    batch_size = config.opt.train_batch_size if is_train else config.opt.val_batch_size
    labels = dataloader.dataset.labels
    patch_overlap = config.dataset.patch_overlap
    sampler_type = config.dataset.sampler_type_train if is_train else config.dataset.sampler_type_val
    label_ratio = config.dataset.label_ratio
    patches_per_brain = config.dataset.patches_per_brain
    val_predictions = {}
    target_metric_name = config.model.target_metric_name if hasattr(config.model,'target_metric_name') else 'dice'

    pov = int(patch_size*patch_overlap) # take high overlap to avoid missing
    if pov%2!=0:
        pov+=1

    if not is_train:
        model.eval()
    else:
        model.train()

    # used to turn on/off gradients
    grad_context = torch.autograd.enable_grad if is_train else torch.no_grad

    with grad_context():

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

            # before augmentation!
            if hasattr(config.dataset, 'predictions_path'):
                assert not config.dataset.shuffle_train # to ensure we got the right label
                pred = torch.load(os.path.join(config.dataset.predictions_path, label)).unsqueeze(0).unsqueeze(0)
                # upsample prediction
                pred = F.interpolate(pred, brain_tensor.shape[2:])
                brain_tensor = torch.cat([brain_tensor, pred], 1) * mask_tensor


            # before augmentation?
            if config.dataset.add_xyz:
                brain_tensor = add_xyz(brain_tensor, mask_tensor)
            
            subject = tio.Subject(t1=tio.ScalarImage(tensor=brain_tensor[0]),
                              mask=tio.LabelMap(tensor=mask_tensor[0]),
                              label=tio.LabelMap(tensor=label_tensor[0])) 
        
            if is_train and (augmentation is not None):
                subject = augmentation(subject)


            if sampler_type == 'grid':
                grid_sampler = tio.inference.GridSampler(subject, patch_size, pov)
                patch_loader = torch.utils.data.DataLoader(grid_sampler, 
                                                           batch_size=patch_batch_size,
                                                           shuffle=True)
                aggregator = tio.inference.GridAggregator(grid_sampler, overlap_mode='average')


            elif sampler_type == 'balanced':
                patch_loader = BalancedSampler(subject, 
                                                patch_size, 
                                                patches_per_brain, 
                                                patch_batch_size, 
                                                label_ratio=label_ratio)

            ########################
            # ITERATE OVER PATCHES #
            ##############################################################################
            metric_dict_patch = defaultdict(list)
            targets_patches = [] # if patch contains fcd
            preds_patches = [] # if fcd predicted in patch
            probs_patches = [] # patch fcd confidence
            hits_patches = [] # hitrate 
            print(f'Iterating for {label}, {len(patch_loader)}')
            for patch_i, patches_batch in enumerate(tqdm(patch_loader)):

                if sampler_type == 'grid':
                    inputs = patches_batch['t1'][tio.DATA].to(device)  # [bs,C,p,p,p]
                    targets = patches_batch['label'][tio.DATA].to(device) # [bs,1,p,p,p]
                else:
                    inputs, targets = patches_batch
                    inputs = inputs.to(device)
                    targets = targets.to(device)

                # with autograd.detect_anomaly():
                with autocast(enabled=config.opt.use_scaler):     
                    logits = model(inputs)

                    # if config.opt.criterion == "BCE":
                    #     loss = compute_BCE(logits, targets, config)
                   
                    # elif config.opt.criterion == "DiceBCE":
                    #     loss = compute_BCE(logits, targets, config)
                    #     loss += DiceLossBinary(logits, targets)
                    
                    # else:
                    
                    loss = criterion(logits, targets) # [bs,1,p,p,p], [bs,1,p,p,p]

                metric_dict_patch[f'{config.opt.criterion}'].append(loss.item())

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

                    metric_dict_patch['grad_norm'].append(calc_gradient_norm(filter(lambda x: x[1].requires_grad, 
                                                  model.named_parameters())))
                    
                    
                    if config.opt.use_scaler:
                        scaler.step(opt)
                        scaler.update()
                    else:
                        opt.step()


                # casting back to patch
                if sampler_type == 'grid':
                    locations = patches_batch[tio.LOCATION]
                    aggregator.add_batch(logits.detach(), locations)

                ####################
                # HIT-RATE METRICS #
                ####################

                targets_sum = targets.sum((-1,-2,-3)) # [bs,1]
                targets_ =  targets_sum > 0
                targets_ = targets_.squeeze(1).detach().cpu().numpy().astype(int)
                
                hits_ = (logits * targets).sum((-1,-2,-3)) > 0.1*targets_sum # hit = 10% intersection
                hits_ = hits_.squeeze(1).detach().cpu().numpy().astype(int)
                probs_ = logits.mean((-1,-2,-3)).squeeze(1).detach().cpu().numpy()
                preds_ = (probs_ > 0.5).astype(int)

                targets_patches.append(targets_)
                hits_patches.append(hits_)
                probs_patches.append(probs_)
                preds_patches.append(preds_)

                #####################
                # per-PATCH METRICS #
                #####################
                coverage = (logits*targets).sum((-1,-2,-3)) / (targets.sum((-1,-2,-3)) + 1e-5)
                dice_score = DiceScoreBinary(logits, targets).item()

                metric_dict_patch[f'coverage'].append(coverage.mean().item())
                metric_dict_patch[f'dice_score'].append(dice_score)

            
            ##############################################################################
            for k,v in metric_dict_patch.items():
                metric_dict[k].append(np.mean(v))

            targets_all=np.concatenate(targets_patches)
            preds_all=np.concatenate(preds_patches)
            probs_all=np.concatenate(probs_patches)
            hits_all=np.concatenate(hits_patches)


            ##################
            # GRID - METRICS #
            ##################
            if sampler_type == 'grid':

                #######################
                # Whole-brain metrics #
                #######################
                output_tensor = aggregator.get_output_tensor().unsqueeze(1) # [1,1,H,W,D]
                output_tensor = output_tensor * mask_tensor # zeros all non mask values
                dice = DiceScoreBinary(output_tensor, label_tensor).item()
                coverage = (output_tensor*label_tensor).sum() / label_tensor.sum()
                metric_dict['dice_score'].append(dice)
                metric_dict['coverage'].append(coverage.item())

                ###########
                # HITRATE #
                ###########
                # sorting by predicted probabilities
                argsort = np.argsort(probs_all, axis=0)[::-1]
                for top_k in [5, 10, 25]:
                    top_k_fcd = targets_all[argsort][:top_k]
                    hitrate = top_k_fcd.mean()
                    metric_dict[f'top-{top_k}_hitrate'].append(hitrate)

                if not is_train:
                    val_predictions[label] = output_tensor.detach().cpu().numpy()

            
            try:
                accuracy = accuracy_score(targets_all, preds_all)
                precision = precision_score(targets_all, preds_all, zero_division=0)
                recall = recall_score(targets_all, preds_all, zero_division=0)
                roc_auc = roc_auc_score(targets_all, probs_all, average='samples')
                hitrate = (hits_all*targets_all).sum() / targets_all.sum()
            except Exception as e:
                print(e,' - error during sklearn metrics calculation!')
                accuracy, precision, recall, roc_auc, hitrate = 0,0,0,0,0
                pass

            metric_dict['accuracy'].append(accuracy)
            metric_dict['precision'].append(precision)
            metric_dict['recall'].append(recall)
            metric_dict['roc_auc_samples'].append(roc_auc)
            metric_dict['hitrate_0.1'].append(hitrate)


            #########
            # PRINT #
            #########
            message = f'For {phase_name}, {label},'
            for title, value in metric_dict.items():
                if title == 'grad_norm_times_lr':
                    v = np.round(value[-1],6)
                else:
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
            # use greedy-saving: save only if the target metric is improved
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
    # essential for the proper samplers functioning and torchio augmentation
    assert config.opt.val_batch_size == 1
    assert config.opt.train_batch_size == 1

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
        model_dict = torch.load(config.model.weights)
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
        noise = tio.RandomNoise(std=(0,1e-2))
        blur = tio.RandomBlur((0,1e-2))
        affine = tio.RandomAffine(scales=(0.95, 1.05, 0.95, 1.05, 0.95, 1.05), 
                                 degrees=3,
                                 translation=(1,1,1),
                                 center='image',
                                 default_pad_value=0)
        rescale = tio.RescaleIntensity(out_min_max=(0, 1))
        augmentation = tio.Compose([symmetry, noise, blur, affine, rescale])

    ################
    # CREATE OPTIM #
    ################
    criterion = {
        "Dice": DiceLossBinary, 
        "BCE": nn.BCELoss,
        "DiceBCE":None,
        "SFL": symmetric_focal_loss(delta=config.opt.delta, gamma=config.opt.gamma),
        "USFL":sym_unified_focal_loss(weight=config.opt.weight, # 0.5
                                         delta=config.opt.delta,  # 0.6
                                         gamma=config.opt.gamma) # 0.5
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
    main(args)
    
