import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import sklearn.metrics as mt
import matplotlib.pyplot as plt

import os.path as path
import copy
import time
import itertools

from dataset import get_loader, get_dataset, LABELS

RESULTS_DIR = './results'

def get_model(pretrained=True, finetune=False, arch="densenet"):
    if arch == "resnet":
        model = models.resnet101(pretrained=pretrained)
    else:
        model = models.densenet121(pretrained=pretrained)
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

    best_model_wts = copy.deepcopy(model.state_dict())
    least_loss = float("inf")

    epoch_losses = { "train": [], "val": [] }
    
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
                loader = itertools.islice(loader, max_samples)
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
                least_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
            epoch_losses[phase].append(epoch_loss)
        print()
        
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Least loss: {:4f}'.format(least_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return {
        "model": model,
        "losses": epoch_losses
    }

def evaluate(model, dataloader, device):
    model.eval()
    all_preds = None
    all_labels = None
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        preds = torch.sigmoid(outputs)
        np_preds = preds.detach().cpu().numpy()
        np_labels = labels.data.cpu().numpy()
        if all_preds is None:
            all_preds = np_preds
            all_labels = np_labels
        else:
            all_preds = np.vstack((all_preds, np_preds))
            all_labels = np.vstack((all_labels, np_labels))
    print("AVG AUC", mt.roc_auc_score(all_labels, all_preds))
    return all_labels, all_preds
            

class Trainer():
    def __init__(self, finetune=False, max_train_samples=None, lr=0.0001, epochs=3, arch="densenet", output_path="./model/model.pth"):
        print("Training using options", "arch", arch, "finetune", finetune)
        self.output_path = output_path
        self.model = get_model(finetune=finetune, arch="densenet")
        params = self.model.parameters() if finetune else self.model.classifier.parameters()
        self.optimizer = optim.Adam(params, lr=lr)
        self.max_train_samples=max_train_samples
        self.criterion = nn.BCEWithLogitsLoss()
        self.dataloaders = {
            "train": get_loader(get_dataset("train")),
            "val": get_loader(get_dataset("val"))
        }
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.epochs=epochs
        self.train_results = None;

    def train(self):
        self.train_results = train_model(
            model=self.model,
            dataloaders=self.dataloaders,
            criterion=self.criterion,
            optimizer=self.optimizer,
            device=self.device,
            num_epochs=self.epochs,
            max_samples=self.max_train_samples
        )
        best_model = self.train_results["model"]
        torch.save(best_model.state_dict(), self.output_path)
        return self.train_results
    
    def evaluate(self):
        model = self.train_results["model"]
        return evaluate(model, get_loader(get_dataset("val")), device=self.device)


def plot_roc_auc(y_true, y_pred, save_to_file=False, prefix=""):
    fpr = {}
    tpr = {}
    roc_auc = {}
    for i in range(len(LABELS)):
        fpr[i], tpr[i], _ = mt.roc_curve(y_true[:, i], y_pred[:, i])
        roc_auc[i] = mt.auc(fpr[i], tpr[i])

        plt.figure()
        lw = 2
        plt.plot(fpr[i], tpr[i], color='darkorange',
                lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[i])
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(LABELS[i])
        plt.legend(loc="lower right")
        if save_to_file:
            filename = f'{prefix}auc_{LABELS[i].replace(" ", "_")}.png'
            plt.savefig(path.join(RESULTS_DIR, filename))
        else:
            plt.show()