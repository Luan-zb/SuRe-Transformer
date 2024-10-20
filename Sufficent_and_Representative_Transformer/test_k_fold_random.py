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
from torchmetrics import Accuracy, Precision, Recall, F1Score, AUROC,Specificity
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import logging
#internal imports
from data import get_multi_cohort_df,MILDataset
from utils.utils import get_loss, get_model, get_optimizer, get_scheduler

parser = argparse.ArgumentParser(description='testing for histopathology images')
parser.add_argument(
    "--config_file",
    help="path of config file",
    default="config/config_Transformer_for_testdataset_k_fold_random.yaml",
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
        
def print_network(net):
    num_params = 0
    num_params_trainable = 0
    for param in net.parameters():
        num_params += param.numel()
        if param.requires_grad:
            num_params_trainable += param.numel()
    print('Total number of parameters: %d' % num_params)
    print('Total number of trainable parameters: %d' % num_params_trainable)

def main(args):
    # set up the paths
    base_path=Path(args.save_dir)
    print("base_path: ", base_path)
    
    base_path.mkdir(parents=True, exist_ok=True)

    model_path=base_path/'models'
    result_path=base_path/'results'
    model_path.mkdir(parents=True, exist_ok=True)
    result_path.mkdir(parents=True, exist_ok=True)

    roc_path=base_path/'figures'
    roc_path.mkdir(parents=True, exist_ok=True)

    # summarywriter by tensorboard, Generate training and testing data under the path
    log_path=base_path/args.log_path
    log_path.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=log_path)

    # load data
    print("\n--- load dataset ---")
    # get the cohort df(PATIENT,TARGET,FILENAME,slide_path)
    data= get_multi_cohort_df(
         args.data_config,
         args.cohorts, [args.target],
         args.label_dict,
         norm=args.norm,
         feats=args.feats,
    )    
    
    train_cohorts = f'{",".join(args.cohorts)}'
    test_cohorts=[ train_cohorts ]

    all_results=[]
    for i in range(args.folds):
        csv_path='{}/splits_{}.csv'.format(args.split_dir, i)
        all_splits = pd.read_csv(csv_path)  
        slide_data = pd.read_csv(args.csv_path)  
        print("all_splits: ", all_splits)
        print("slide_data: ", slide_data)
        
        # create the dataset for cross_validation
        test_split_ids = get_split_from_df(slide_data, all_splits, 'test')
        print("test_split_ids:",test_split_ids)
        print(f'num test samples in fold {i}: {len(test_split_ids)}')
        
        torch.manual_seed(42)
        test_dataset = MILDataset(
            data,
            test_split_ids, [args.target],
            norm=args.norm
        )
        test_dataloader = DataLoader(test_dataset, batch_size=args.bs)

        model= get_model(
            args.model,
            num_classes=args.num_classes,
            input_dim=args.input_dim,
            **args.model_config
        )

        print_network(model)
        # --------------------------------------------------------
        # testing
        # --------------------------------------------------------
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)   
        criterion= get_loss(args.criterion, pos_weight=args.pos_weight) if args.task == "binary" else get_loss(args.criterion)
        criterion=criterion.to(device)
        print("criterion:",criterion)
        print("criterion setting finished!")

        filename=f'model_fold{i}.pth'
        model_path_name=os.path.join(model_path,filename)
        state_dict=torch.load(model_path_name)
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict,strict=False)            
        fold_results=test(model, test_dataloader,writer,device,criterion,roc_path,i,args)
        all_results.append(fold_results)
    # After running all folds, save all_results to a CSV file
    results_df = pd.DataFrame(all_results)
    results_csv_path = os.path.join(result_path, "all_results.csv")
    results_df.to_csv(results_csv_path, index=False)
        

# Test script
def test(model, test_loader, writer, device,criterion,roc_path,fold,args):
    model.eval()
    accuracy = Accuracy(task=args.task, num_classes=args.num_classes).to(device)
    precision = Precision(task=args.task, average='macro', num_classes=args.num_classes).to(device)
    recall = Recall(task=args.task, average='macro', num_classes=args.num_classes).to(device)
    auroc = AUROC(task=args.task).to(device)
    f1 = F1Score(num_classes=args.num_classes, task=args.task, average='macro').to(device)
    specificity=Specificity(num_classes=args.num_classes, task=args.task, average='macro').to(device)

    pred_scores = [] 
    true_labels = []
    test_loss=0
    correct = 0
    total = 0
    fold_results={}
    with torch.no_grad():
        for images, coords, labels, _, _ ,distances_matrix in test_loader:
            images = images.to(device)
            labels = labels.to(device) 
            distances_matrix = distances_matrix.to(device)
            outputs = model(images,coords,distances_matrix)
            loss= criterion(outputs,labels)
            _, predicted = torch.max(outputs.data, 1)
            accuracy(predicted, labels.data)
            precision(predicted, labels.data)
            recall(predicted, labels.data)
            f1(predicted, labels.data)
            auroc(predicted, labels.data)
            specificity(predicted, labels.data)

            pred_scores.extend(outputs[:, 1].cpu().numpy()) 
            true_labels.extend(labels.cpu().numpy())
    acc = accuracy.compute().item() 
    prec = precision.compute().item() 
    rec = recall.compute().item() 
    f1_score = f1.compute().item()
    auroc_score = auroc.compute().item()
    spec=specificity.compute().item()

    fold_results['fold']=fold
    fold_results['accuracy'] = acc
    fold_results['precision'] = prec
    fold_results['recall'] = rec
    fold_results['f1_score'] = f1_score
    fold_results['auroc_score'] = auroc_score
    fold_results['specificity'] = spec
    
    logging.info(f"Test Accuracy: {acc:.4f}, Test precision: {prec:.4f}, Test recall: {rec:.4f}, Test f1: {f1_score:.4f}, Test auroc: {auroc_score:4f},Test specificity:{spec:.4f}")
    logging.error("This is a fatal log!")   
    fpr, tpr, thresholds = roc_curve(true_labels, pred_scores)
    auc_value = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f'AUC = {auc_value:.2f}')
    plt.plot([0, 1], [0, 1], 'k--')  
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(roc_path,"roc.png"))
    return fold_results
    
    
    
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
