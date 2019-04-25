import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

import os.path as path

DATA_DIR = './data'
TRAIN_CSV = './data/CheXpert-v1.0-small/train.csv'
VALID_CSV = './data/CheXpert-v1.0-small/valid.csv'

LABELS = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']
BATCH_SIZE = 4

def data_path(file_path):
    return path.join(DATA_DIR, file_path)


class TrainingDataset(Dataset):
    def __init__(self, csv_file, data_dir, transform=None, value_for_uncertain=1.0):
        self.df = pd.read_csv(csv_file)
        self.data_dir = data_dir
        self.transform = transform
        self.value_for_uncertain = value_for_uncertain
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_path = path.join(self.data_dir, self.df.iloc[idx]['Path'])
        img = Image.open(img_path).convert('RGB')
        labels = self.df.iloc[idx][LABELS].astype(np.float32).replace(np.NaN, 0.0).replace(-1.0, self.value_for_uncertain).values
        labels = torch.from_numpy(labels)
        if self.transform:
            img = self.transform(img)
        return img, labels

class TestDataset(Dataset):
    def __init__(self, csv_file, transform=None, data_dir=None):
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        self.data_dir = data_dir
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]['Path']
        if self.data_dir is not None:
            img_path = path.join(self.data_dir, img_path)
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, img_path

def get_transformer():
    return transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def get_dataset(train_or_val, value_for_uncertain=1.0):
    csv_file = TRAIN_CSV if train_or_val == "train" else VALID_CSV
    return TrainingDataset(csv_file, DATA_DIR, get_transformer(), value_for_uncertain=value_for_uncertain)

def get_loader(dataset, shuffle=True):
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=4)

def get_train_loader(value_for_uncertain=1.0):
    return get_loader((get_dataset('train'), value_for_uncertain))

def get_val_loader():
    return get_loader((get_dataset('val')), shuffle=False)

def get_test_dataset(csv_file, data_dir=None):
    return TestDataset(csv_file, get_transformer(), data_dir=data_dir)

def get_test_loader(csv_file, data_dir=None):
    return get_loader(get_test_dataset(csv_file, data_dir=data_dir), shuffle=False)

