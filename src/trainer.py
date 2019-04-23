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

from dataset import get_loader, get_dataset, LABELS, BATCH_SIZE
from model import get_model, get_optimizer, load_ensemble_from_checkpoints
from util import CheckpointManager

RESULTS_DIR = './results'
CHECKPOINT_COUNT = 1#4800
CHECKPOINTS_PER_RUN = 2#10

def train_model(model, dataloaders, criterion, optimizer, device, num_epochs=3, max_samples=None):
    since = time.time()
    least_loss = float("inf")
    epoch_losses = { "train": [], "val": [] }
    checkpoints = CheckpointManager(max_items=CHECKPOINTS_PER_RUN)
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        
        # each epoch has training and validation phase
        phase_results = train_epoch(
            model=model,
            dataloaders=dataloaders,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            checkpoints=checkpoints,
            max_samples=max_samples,
            time_started=since
        )
        epoch_loss = phase_results["epoch_loss"]
        epoch_losses['train'].append(epoch_loss)
        if epoch_loss < least_loss:
            least_loss = epoch_loss
        print()
        
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Least loss: {:4f}'.format(least_loss))
    return {
        "losses": epoch_losses,
        "checkpoints": checkpoints
    }

def train_epoch(model, dataloaders, criterion, optimizer, device, checkpoints, time_started, max_samples=None):
    model.train()
    running_loss = 0.0
    train_loader = dataloaders['train']
    val_loader = dataloaders['val']
    data_size = len(train_loader.dataset)
    if max_samples is not None:
        train_loader = itertools.islice(train_loader, max_samples)
        data_size = max_samples
    for i, (inputs, labels) in enumerate(train_loader):
        batch = i + 1
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

        if batch % CHECKPOINT_COUNT == 0:
            # evaluate batch
            batch_results = evaluate(model=model, dataloader=val_loader, device=device, criterion=criterion)
            eval_labels, eval_preds, eval_loss = (
                batch_results['labels'], batch_results['predictions'], batch_results['loss']
            )
            avg_auc = mt.roc_auc_score(eval_labels, eval_preds)
            checkpoints.add(model, loss.item(), avg_auc, eval_loss)
            elapsed = time.time() - time_started
            print('Batch', batch, 'Batch loss', loss.item(), 'Val AUC', avg_auc, 'Val Loss', eval_loss, "Elapsed", elapsed)
    
    epoch_loss = running_loss / data_size
    print('{} Loss: {:.4f}'.format(
        'Training', epoch_loss))
    
    return {
        "epoch_loss": epoch_loss
    }

def evaluate(model, dataloader, device, criterion=None):
    model.eval()
    running_loss = 0.0
    all_preds = None
    all_labels = None
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            preds = torch.sigmoid(outputs)
            if criterion is not None:
                loss = criterion(outputs, labels)
                running_loss += loss.item() * inputs.size(0)
        np_preds = preds.detach().cpu().numpy()
        np_labels = labels.data.cpu().numpy()
        if all_preds is None:
            all_preds = np_preds
            all_labels = np_labels
        else:
            all_preds = np.vstack((all_preds, np_preds))
            all_labels = np.vstack((all_labels, np_labels))
    
    epoch_loss = None if criterion is None else running_loss / len(dataloader.dataset)
    return {
        "labels": all_labels,
        "predictions": all_preds,
        "loss": epoch_loss
    }
            

class Trainer():
    def __init__(self, finetune=False, max_train_samples=None, lr=0.0001, epochs=3, arch="densenet", output_path="./models/model"):
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
        self.train_results = None

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
        checkpoints = self.train_results["checkpoints"]
        checkpoints.save(self.output_path)
        return self.train_results
    
    def evaluate(self):
        model = load_ensemble_from_checkpoints(self.train_results["checkpoints"].checkpoints)
        results = evaluate(model, get_loader(get_dataset("val")), device=self.device)
        labels, preds = results["labels"], results["predictions"]
        print("AVG AUC", mt.roc_auc_score(labels, preds))
        return labels, preds


def plot_roc_auc(y_true, y_pred, save_to_file=False, prefix="", output_path=RESULTS_DIR):
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
            plt.savefig(path.join(output_path, filename))
        else:
            plt.show()