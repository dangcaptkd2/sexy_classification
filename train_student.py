import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.transforms.functional as F

from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import os
import mlflow
import yaml
from yaml.loader import SafeLoader

import timm
from timm.loss import LabelSmoothingCrossEntropy

from optimizer.adan import Adan
from optimizer.sam import SAM
from optimizer.ranger21.ranger21 import Ranger21
from utils import train_model, evaluate_model, train_student_model
from dataset import ImageStudentTeacher, get_normal_dataloader

import argparse


def get_dataloader(cfg, test=False):
    mode = ['train', 'val'] if not test else ['test']
    image_datasets = {x: ImageStudentTeacher(path_data=cfg['data_dir'], mode=x) 
                        for x in mode}
    shuffle_dict = {'train': True, 'val': False, 'test': False}
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=cfg['batch_size'], shuffle=shuffle_dict[x], num_workers=2) 
                        for x in mode}

    dataset_sizes = {x: len(image_datasets[x]) for x in mode}
    return dataloaders, dataset_sizes

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--who', default='root')
    args = parser.parse_args()

    with open(f"./configs/{args.who}.yaml") as f:
        cfg = yaml.load(f, Loader=SafeLoader)

    model = timm.create_model(cfg['model_name'], pretrained=True, num_classes=3)
    print(f"Number parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)} params")
    model.to(cfg['device'])

    if args.eval:
        # dataloaders, dataset_sizes = get_dataloader(cfg, test=True)
        dataloaders, dataset_sizes, class_names = get_normal_dataloader(cfg)
        model.load_state_dict(torch.load(cfg['model_path']))
        evaluate_model(model, dataloaders, cfg)
    else:
        dataloaders, dataset_sizes = get_dataloader(cfg)
        if cfg['model_resume'] and cfg['model_path'] is not None:
            model.load_state_dict(torch.load(cfg['model_path']))
        print("Total train step:", cfg['num_epochs']*len(dataloaders['train']))

        if cfg['loss'] == 'CrossEntropy':
            criterion = nn.CrossEntropyLoss()
        if cfg['loss'] == 'LabelSmoothing':
            criterion = LabelSmoothingCrossEntropy()

        if cfg['optimizer'] == "Adam":
            optimizer = optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=1e-5)
        elif cfg['optimizer'] == "Adan":
            optimizer = Adan(model.parameters(), lr=cfg['lr'], weight_decay=1e-5)
        if cfg['optimizer'] == "sam":
            optimizer = SAM(model.parameters(), lr=cfg['lr'], weight_decay=1e-5)

        scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

        with mlflow.start_run(run_name=cfg['run_name'], 
                            description="First stage training sexy model with biggest dataset (datset_0)"):
            mlflow.log_param("Name dataset", cfg['data_name'])
            mlflow.log_param("Name model", cfg['model_name'])
            trained_model, best_acc = train_student_model(model, dataloaders, criterion, optimizer, scheduler, dataset_sizes, 
                                                    cfg=cfg)
            

        best_ckp_path = f"./ckps/{cfg['name']}_{cfg['model_name']}_ckps_{best_acc}.pt"
        torch.save(trained_model.state_dict(), best_ckp_path)
        mlflow.log_artifact(best_ckp_path)
        mlflow.end_run()