import torch
import torch.utils.data as data
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torchvision.transforms.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import Sampler,RandomSampler,SequentialSampler

import os
import cv2
import numpy as np
from PIL import Image
import pandas as pd

import random
from auto_augment import rand_augment_transform

_FILL = (128, 128, 128)
_IMAGENET_DEFAULT_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_DEFAULT_STD = [0.229, 0.224, 0.225]

tfm = rand_augment_transform(
    config_str='rand-m27-n2-mstd0.5', 
    hparams={}
)

class SquarePad:
    def __call__(self, image):
        w, h = image.size
        max_wh = np.max([w, h])
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = (hp, vp, hp, vp)
        return F.pad(image, padding, _FILL, 'constant')

def get_normal_dataloader(cfg):
    data_transforms = {
            'train': transforms.Compose([
                SquarePad(),
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(_IMAGENET_DEFAULT_MEAN, _IMAGENET_DEFAULT_STD)
            ]),
            'val': transforms.Compose([
                SquarePad(),
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(_IMAGENET_DEFAULT_MEAN, _IMAGENET_DEFAULT_STD)
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


class ImageStudentTeacher(data.Dataset):
    """
    Helper Class to create the pytorch dataset
    """
    t_transforms = transforms.Compose([
            SquarePad(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(_IMAGENET_DEFAULT_MEAN, _IMAGENET_DEFAULT_STD) 
        ])
    s_transforms = t_transforms
    s_transforms.transforms.insert(0, tfm)

    def __init__(self, path_data, mode='train') -> None:
        super().__init__()
        self.mode = mode
        
        df = pd.read_csv(path_data)
        self.df_teacher = df[df['who']=='real']
        self.df_student = df[df['who']=='fake']
        self._clean_dataframe()
        if mode=='test':
            self.df = pd.concat([self.df_teacher, self.df_student]).reset_index(drop=True)
        else:
            self.df_teacher = self.df_teacher[self.df_teacher['mode']==self.mode]
            self.df_student = self.df_student[self.df_student['mode']==self.mode]
    
    def _clean_dataframe(self):
        self.df_student = self.df_student[self.df_student['score']>=0.7].reset_index(drop=True)
        n = int(len(self.df_student)/len(self.df_teacher))
        m = len(self.df_student)%len(self.df_teacher)
        self.df_teacher = pd.concat([self.df_teacher]*n+[self.df_teacher[:m]]).reset_index(drop=True)
        self.df_teacher = self.df_teacher.drop(['score', 'who'], axis=1)
        self.df_student = self.df_student.drop(['score', 'who'], axis=1)
        assert len(self.df_student) == len(self.df_teacher), "student and teacher do not have the same length"

        self.df_teacher = self._split(self.df_teacher)
        self.df_student = self._split(self.df_student)

    def _split(self, df):
        train, val = train_test_split(df, test_size=0.2, random_state=42)
        train['mode'] = 'train'
        val['mode'] = 'val'
        return pd.concat([train, val]).reset_index(drop=True)

    def __getitem__(self, index):
        if self.mode=='test':
            row = self.df.iloc[index]
            label = row.label
            img = Image.open(row.path)
            img = self.s_transforms(img)
            return img, torch.tensor(label)

        row_t = self.df_teacher.iloc[index]
        row_s = self.df_student.iloc[index]

        label_t = row_t.label
        label_s = row_s.label
        img_t = Image.open(row_t.path)
        img_s = Image.open(row_s.path)

        img_t = self.t_transforms(img_t)
        img_s = self.s_transforms(img_s)

        return img_t, torch.tensor(label_t), img_s, torch.tensor(label_s)
    
    def __len__(self):
        return len(self.df_student)

if __name__ == '__main__':
    loader = ImageStudentTeacher(path_data='/quyennt/sexy/data/pseudo_label/pseudo_label_0.csv')
    print(len(loader))
    for a,b,c,d in loader:
        print(a.shape)
        print(b)
        print(c.shape)
        print(d)
        break