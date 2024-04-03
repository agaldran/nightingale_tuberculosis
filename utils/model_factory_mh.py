import sys
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import swin_t, Swin_T_Weights

def get_model(model_name, n_classes=2, dropout=0.):
    if model_name == 'resnext50':
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        num_ftrs = model.fc.in_features
        linear_cl = torch.nn.Linear(num_ftrs, n_classes)
        if dropout > 0:
            model.fc = nn.Sequential(nn.Dropout(p=dropout), linear_cl)
        else: model.fc = linear_cl

    elif model_name == 'swin_t':
        model = swin_t(weights=Swin_T_Weights)
        num_ftrs = model.head.in_features
        linear_cl = nn.Linear(in_features=num_ftrs, out_features=n_classes)
        if dropout > 0:
            model.head = nn.Sequential(nn.Dropout(p=dropout), linear_cl)
        else: model.head = linear_cl
    else:
        sys.exit('{} is not a valid model_name, check utils.model_factory_mh.py'.format(model_name))

    setattr(model, 'n_classes', n_classes)
    return model



