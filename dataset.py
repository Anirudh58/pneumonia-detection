import os
from PIL import Image

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

target_labels = ['Normal', 'Pnemonia']

# create a map for label->id
label_id_map = {}
id_label_map = {}

for i in range(0 , len(target_labels)):
    label_id_map[target_labels[i]] = i
    id_label_map[i] = target_labels[i]

class XRayDataset(Dataset):
    def __init__(self, mode, dataset_path, metadata_path, transforms=None):
        self.dataset_path = dataset_path
        self.mode = mode
        self.transforms = transforms
        df_dataset = pd.read_csv(metadata_path)
        df_dataset = df_dataset.loc[:, ~df_dataset.columns.str.contains('^Unnamed')]
        
        if self.mode == 'train':
            self.dataset = df_dataset[df_dataset['Dataset_type'] == 'TRAIN']
        elif self.mode == 'test':
            self.dataset = df_dataset[df_dataset['Dataset_type'] == 'TEST']
            
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        
        img_name = self.dataset.iloc[idx, 0]
        img_path = os.path.join(self.dataset_path, self.mode, img_name)
        
        img = Image.open(img_path).convert('RGB')
        #img = Image.open(img_path)
        
        image_id = torch.tensor([idx])
        
        target = {}
        
        target['label_1'] = label_id_map[self.dataset.iloc[idx, 1]]
        target['image_id'] = image_id
        #target['label_2'] = self.dataset.iloc[idx, 3] if isinstance(self.dataset.iloc[idx, 3], str) else 'None'
        #target['label_3'] = self.dataset.iloc[idx, 4] if isinstance(self.dataset.iloc[idx, 4], str) else 'None'
        
        if self.transforms is not None:
            img = self.transforms(img)
        
        return img, target
            
        
    