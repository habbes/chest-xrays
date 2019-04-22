import torch.nn as nn
import torch.optim as optim
from torchvision import models
from dataset import LABELS

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