import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import os

from dataset import LABELS
from util import get_device

def get_model(pretrained=True, finetune=False, arch="densenet", layers=18):
    if arch == "resnet":
        return get_resnet_model(pretrained=pretrained, finetune=finetune, layers=layers)
    else:
        return get_densenet_model(pretrained, finetune)

def get_densenet_model(pretrained=True, finetune=False):
    model = models.densenet121(pretrained=pretrained)
    if not finetune:
        for param in model.parameters():
            param.requires_grad = False
    in_features = model.classifier.in_features
    out_features = len(LABELS)
    model.classifier = nn.Linear(in_features, out_features)
    return model

def get_resnet_model(pretrained=True, finetune=False, layers=18):
    kwargs = { "pretrained": pretrained }
    if layers == 18:
        model = models.resnet18(**kwargs)
    elif layers == 34:
        model = models.resnet34(**kwargs)
    elif layers == 50:
        model = models.resnet50(**kwargs)
    else:
        raise f"Invalid Resnet type {layers}"
    if not finetune:
        for param in model.parameters():
            param.requires_grad = False
    in_features = model.cf.in_features
    out_features = len(LABELS)
    model.fc = nn.Linear(in_features, out_features)

def get_optimizer(params, **kwargs):
    return optim.Adam(params, **kwargs)

class Ensemble(nn.Module):
    """
    Ensemble model, averages outputs
    from all the individual models.
    Should only be used for inference
    """
    def __init__(self, models):
        super(Ensemble, self).__init__()
        for i, model in enumerate(models):
            self.add_module(f'model{i}', model)
    
    def forward(self, x):
        ens = None
        num_modules = 0
        for model in self.modules():
            num_modules += 1
            y = model(x)
            y = torch.sigmoid(y)
            ens = y if ens is None else torch.cat((ens, y), dim=-1)
        ens = ens.view(-1, num_modules, len(LABELS))
        ens = ens.mean(dim=1)
        return ens


def load_models(model_dirs):
    """
    load models from all the provided directories.
    Any file with a .pth extension is considered a model file.
    """
    models = []
    for directory in model_dirs:
        filenames = os.listdir(directory)
        for filename in filenames:
            if filename.endswith('.pth'):
                model_path = os.path.join(directory, filename)
                model = get_model()
                model.load_state_dict(torch.load(model_path))
                models.append(model)
    return models

def load_ensemble_from_dirs(model_dirs):
    """
    Create ensemble classifier from models
    stored in the specified directories
    """
    models = load_models(model_dirs)
    ensemble = Ensemble(models)
    ensemble.to(get_device())
    return ensemble

def load_ensemble_from_checkpoints(checkpoints):
    models = []
    for chk in checkpoints:
        model = get_model()
        model.load_state_dict(chk["model"])
        models.append(model)
    ensemble = Ensemble(models)
    ensemble.to(get_device())
    return ensemble