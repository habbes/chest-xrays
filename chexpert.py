import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, models
from PIL import Image

import os.path as path
import copy
import time
import itertools

DATA_DIR = './data'
TRAIN_CSV = './data/CheXpert-v1.0-small/train.csv'
VALID_CSV = './data/CheXpert-v1.0-small/valid.csv'

LABELS = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']

def data_path(file_path):
    return path.join(DATA_DIR, file_path)


class CheXpertDataset(Dataset):
    def __init__(self, csv_file, data_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.data_dir = data_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_path = path.join(self.data_dir, self.df.iloc[idx]['Path'])
        img = Image.open(img_path).convert('RGB')
        labels = self.df.iloc[idx][LABELS].astype(np.float32).replace(np.NaN, 0.0).replace(-1.0, 0.0).values
        labels = torch.from_numpy(labels)
        if self.transform:
            img = self.transform(img)
        return img, labels

def get_transformer():
    return transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def get_dataset(train_or_val):
    csv_file = TRAIN_CSV if train_or_val == "train" else VALID_CSV
    return CheXpertDataset(csv_file, DATA_DIR, get_transformer())

def get_loader(dataset, shuffle=True):
    return DataLoader(dataset, batch_size=64, shuffle=shuffle, num_workers=4)

def get_train_loader():
    return get_loader((get_dataset('train')))

def get_val_loader():
    return get_loader((get_dataset('val')), shuffle=False)

def get_model(pretrained=True, finetune=False):
    model = models.densenet121(pretrained=True)
    if not finetune:
        for param in model.parameters():
            param.requires_grad = False
    in_features = model.classifier.in_features
    out_features = len(LABELS)
    model.classifier = nn.Linear(in_features, out_features)
    return model

def get_optimizer(params, **kwargs):
    return optim.Adam(params, **kwargs)

def train_model(model, dataloaders, criterion, optimizer, device, num_epochs=3, max_samples=None):
    since = time.time()
    val_acc_history = []
    
    best_model_wts = copy.deepcopy(model.state_dict())
    least_loss = float("inf")
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        
        
        # each epoch has training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            
            
            running_loss = 0.0
            
            
            # iterate over data
            loader = dataloaders[phase]
            data_size = len(loader.dataset)
            if phase == "train" and max_samples is not None:
                loader = itertools.islice(data_size, max_samples)
                data_size = max_samples
            for inputs, labels in loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # reset optimizer gradients
                optimizer.zero_grad()
                
                
                # forward pass
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    _, preds = torch.max(outputs, 1)
                    
                    # backward pass if training
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # statistics
                running_loss += loss.item() * inputs.size(0)
            
            epoch_loss = running_loss / data_size

            print('{} Loss: {:.4f}'.format(
                phase, epoch_loss))
            
            # keep copy if best model so far
            if phase == 'val' and epoch_loss < least_loss:
                least_loss = least_loss
                best_model_wts = copy.deepcopy(model.state_dict())
            
        print()
        
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Least loss: {:4f}'.format(least_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

class Trainer():
    def __init__(self, finetune=False, max_train_samples=None, lr=0.0001, epochs=3):
        self.model = get_model(finetune=finetune)
        params = self.model.parameters() if finetune else self.model.classifier.parameters()
        self.optimizer = optim.Adam(params, lr=lr)
        self.max_train_samples=None
        self.criterion = nn.BCEWithLogitsLoss()
        self.dataloaders = {
            "train": get_loader(get_dataset("train")),
            "val": get_loader(get_dataset("val"))
        }
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.epochs=epochs
    
    def train(self):
        return train_model(
            model=self.model,
            dataloaders=self.dataloaders,
            criterion=self.criterion,
            optimizer=self.optimizer,
            device=self.device,
            num_epochs=self.epochs,
            max_samples=self.max_train_samples
        )
