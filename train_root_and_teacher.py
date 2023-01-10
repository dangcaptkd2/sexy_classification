import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.transforms.functional as F

import numpy as np
import os
import timm
import mlflow
import yaml
from yaml.loader import SafeLoader

from sklearn.utils.class_weight import compute_class_weight
from utils import train_model, evaluate_model

import argparse

class SquarePad:
    def __call__(self, image):
        w, h = image.size
        max_wh = np.max([w, h])
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = (hp, vp, hp, vp)
        return F.pad(image, padding, 0, 'constant')

def get_dataloader(cfg):
    data_transforms = {
            'train': transforms.Compose([
                SquarePad(),
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                SquarePad(),
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
    shuffle_dict = {'train': True, 'val': False}
    image_datasets = {x: datasets.ImageFolder(os.path.join(cfg['data_dir'], x), data_transforms[x]) 
                        for x in ['train', 'val']}
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=cfg['batch_size'], shuffle=shuffle_dict[x], num_workers=2) 
                        for x in ['train', 'val']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    return dataloaders, dataset_sizes, class_names

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--who', default='root')
    args = parser.parse_args()

    with open(f"./configs/{args.who}.yaml") as f:
        cfg = yaml.load(f, Loader=SafeLoader)
    
    dataloaders, dataset_sizes, class_names = get_dataloader(cfg)

    model = timm.create_model(cfg['model_name'], pretrained=True, num_classes=len(class_names))
    print(f"Number parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)} params")
    model.to(cfg['device'])

    if args.eval:
        model.load_state_dict(torch.load(cfg['model_path']))
        evaluate_model(model, dataloaders, cfg)
    else:
        if cfg['model_resume'] and cfg['model_path'] is not None:
            model.load_state_dict(torch.load(cfg['model_path']))
        print('Class names:', class_names)
        print("Total train step:", cfg['num_epochs']*len(dataloaders['train']))

        class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(dataloaders['train'].dataset.targets), y=np.array(dataloaders['train'].dataset.targets))
        class_weights = torch.tensor(class_weights,dtype=torch.float).to(cfg['device'])

        criterion = nn.CrossEntropyLoss(weight=class_weights, reduction='mean')
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        with mlflow.start_run(run_name=cfg['run_name'], 
                            description="First stage training sexy model with biggest dataset (datset_0)"):
            mlflow.log_param("Name dataset", cfg['data_name'])
            mlflow.log_param("Name model", cfg['model_name'])
            trained_model, best_acc = train_model(model, dataloaders, criterion, optimizer, scheduler, dataset_sizes, 
                                                    cfg=cfg)
            

        best_ckp_path = f"./ckps/{cfg['name']}_{cfg['model_name']}_ckps_{best_acc}.pt"
        torch.save(trained_model.state_dict(), best_ckp_path)
        mlflow.log_artifact(best_ckp_path)
        mlflow.end_run()