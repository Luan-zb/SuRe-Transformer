#!/usr/bin/env python
# -*- coding: utf-8 -*-
#external imports
import argparse
import yaml
import os
import torch
import torch.distributed as dist
from pathlib import Path
import pandas as pd
import h5py
from torch.utils.data import DataLoader
import numpy as np
import warnings
from tensorboardX import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
#internal imports
from data import get_multi_cohort_df,MILDataset
from utils.core_utils_k_fold import train,validate
from utils.utils import get_loss, get_model, get_optimizer, get_scheduler

num_gpus = 1
if torch.cuda.is_available() :
    num_gpus = torch.cuda.device_count()
rank = int(os.environ["SLURM_PROCID"])
world_size = int(os.environ["SLURM_NTASKS"])
local_rank = int(os.environ['SLURM_LOCALID'])
os.environ["WORLD_SIZE"] = str(world_size)
os.environ["LOCAL_RANK"] = str(rank % num_gpus)
os.environ["RANK"] = str(rank)
torch.cuda.set_device(local_rank)
dist.init_process_group(backend='nccl')
device = torch.device("cuda", local_rank)
print("rank:", rank)
print("local rank:", local_rank)
print("world_size:", world_size)

parser = argparse.ArgumentParser(description='training for histopathology images')
parser.add_argument(
    "--config_file",
    help="path of config file",
    default="config/config_Transformer_k_fold_random.yaml",
    type=str)

# filter out UserWarnings from the torchmetrics package
warnings.filterwarnings("ignore", category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

def get_split_from_df(slide_data, all_splits=None, split_key='train', return_ids_only=True, split=None): 
    if split is None:
        split = all_splits[split_key]
        split = split.dropna().reset_index(drop=True)  
    if len(split) > 0:
        mask = slide_data['slide_id'].isin(split.tolist())  
        if return_ids_only:
            ids = np.where(mask)[0]
            return ids

def main(args):
    # set up the paths
    base_path=Path(args.save_dir)
    print("base_path: ", base_path)
    
    base_path.mkdir(parents=True, exist_ok=True)
    model_path=base_path/'models'
    result_path=base_path/'results'
    model_path.mkdir(parents=True, exist_ok=True)
    result_path.mkdir(parents=True, exist_ok=True)
    

    
    # load data
    print("\n--- load dataset ---")
    data= get_multi_cohort_df(
        args.data_config,
        args.cohorts, [args.target],
        args.label_dict,
        norm=args.norm,
        feats=args.feats,
    )    
    train_cohorts = f'{",".join(args.cohorts)}'
    val_cohorts=[ train_cohorts ]
    results_validation={t:[] for t in val_cohorts}

    for i in range(args.folds):
        # summarywriter by tensorboard, Generate training and testing data under the path
        log_path=base_path/args.log_path/f'fold_{i}'
        log_path.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=log_path)

        csv_path='{}/splits_{}.csv'.format(args.split_dir, i)
        all_splits = pd.read_csv(csv_path)  #splits_0.csv
        slide_data = pd.read_csv(args.csv_path)  #data_csv/data.csv
        print("all_splits: ", all_splits)
        print("slide_data: ", slide_data)
        
#        # create the dataset for cross_validation
        train_split_ids = get_split_from_df(slide_data, all_splits, 'train')  #return the object
        val_split_ids = get_split_from_df(slide_data, all_splits, 'val')
        test_split_ids = get_split_from_df(slide_data, all_splits, 'test')
        print("train_split_ids: ", train_split_ids)
        print("val_split_ids:",val_split_ids)
        print("test_split_ids:",test_split_ids)
        print(f'num training samples in fold {i}: {len(train_split_ids)}')
        print(f'num validation samples in fold {i}: {len(val_split_ids)}')
        print(f'num test samples in fold {i}: {len(test_split_ids)}')
        
        train_dataset = MILDataset(
            data,
            train_split_ids, [args.target],
            norm=args.norm
        )
        # validation dataset
        val_dataset = MILDataset(
            data, 
            val_split_ids, [args.target],
            norm=args.norm
        )

        # DDP setting
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)

        # DDP setting
        train_dataloader = DataLoader(train_dataset, batch_size=args.bs,  sampler=train_sampler)
        val_dataloader = DataLoader(val_dataset, batch_size=args.bs,  sampler=val_sampler)
    
        # set the val_check_interval
        if len(train_dataset) < args.val_check_interval:
            args.val_check_interval = len(train_dataset)

        # class weighting for binary classification
        if args.task == 'binary':
            df=data.loc[train_split_ids]
            selected_rows = df[df['TARGET'] == 1]
            args.pos_weight = torch.Tensor([len(selected_rows) / len(df)]) 
            print("args.pos_weight: ", args.pos_weight)
        
        model= get_model(
            args.model,
            num_classes=args.num_classes,
            input_dim=args.input_dim,
            **args.model_config
        )
        print(model)
        # DDP setting
        device = torch.device("cuda", local_rank)  
        model = model.to(local_rank)   
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)   
        print("model ddp setting finished!")
        criterion= get_loss(args.criterion, pos_weight=args.pos_weight) if args.task == "binary" else get_loss(config.criterion)
        print("criterion:",criterion)
        print("criterion setting finished!")
        optimizer=get_optimizer(
            name=args.optimizer,
            model= model,
            lr=args.lr,
            wd=args.wd,
        )
        if args.lr_scheduler == 'OneCycleLR':
            args.lr_scheduler_config['total_steps'] = args.num_epochs * len(train_dataloader)
        elif args.lr_scheduler == 'MultiStepLR':
            args.lr_scheduler_config['milestones']=args.milestones
            args.lr_scheduler_config['gamma']=args.gamma
        elif args.lr_scheduler == 'CosineAnnealingWarmRestarts':
            args.lr_scheduler_config['scheduler_periods']=args.scheduler_periods
            args.lr_scheduler_config['scheduler_restart_weights']=args.scheduler_restart_weights
            args.lr_scheduler_config['scheduler_eta_min']=args.scheduler_eta_min
        elif args.lr_scheduler == 'StepLR':
            args.lr_scheduler_config['step_size']=args.step_size
            args.lr_scheduler_config['gamma']=args.gamma

        print("optimizer:",optimizer)
        if args.lr_scheduler:
            scheduler = get_scheduler(
                args.lr_scheduler,
                optimizer,
                **args.lr_scheduler_config
        )
        print("scheduler:",scheduler)
        print("training_validating!")
        
        train(device, model, train_dataloader, val_dataloader, criterion, optimizer,scheduler,writer,model_path,args,i)

        
if __name__=="__main__":
    args=parser.parse_args()
    #load config file from yaml file
    with open(args.config_file, "r") as f:
        config = yaml.safe_load(f)
    
    print("--load config file--")
    for name,value in sorted(config.items()):
        print(f"{name}: {value}")
    
    config=argparse.Namespace(**config)
    main(config)  
