import sys
import torch
import os.path as osp
import torch.nn as nn

from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models import resnet34, ResNet34_Weights
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models.convnext import LayerNorm2d
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from torchvision.models import swin_t, Swin_T_Weights
from torchvision.models import resnext50_32x4d

def get_model(model_name, n_classes=2, n_heads=1, dropout=0.):
    weights_path = '/home/ngsci/vision_weights'
    if model_name == 'resnet18':
        # model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        model = resnet18()
        model.load_state_dict(torch.load(osp.join(weights_path, 'resnet18-f37072fd.pth')))
        num_ftrs = model.fc.in_features
        linear_cl = torch.nn.Linear(num_ftrs, n_classes)
        if dropout > 0:
            model.fc = nn.Sequential(nn.Dropout(p=dropout), linear_cl)
        else: model.fc = linear_cl
    elif model_name == 'resnet34':
        model = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        num_ftrs = model.fc.in_features
        linear_cl = torch.nn.Linear(num_ftrs, n_classes)
        if dropout > 0:
            model.fc = nn.Sequential(nn.Dropout(p=dropout), linear_cl)
        else: model.fc = linear_cl
    elif model_name == 'resnet50':
        # model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        model = resnet50()
        model.load_state_dict(torch.load(osp.join(weights_path, 'resnet50-11ad3fa6.pth')))
        num_ftrs = model.fc.in_features
        linear_cl = torch.nn.Linear(num_ftrs, n_classes)
        if dropout > 0:
            model.fc = nn.Sequential(nn.Dropout(p=dropout), linear_cl)
        else: model.fc = linear_cl
    elif model_name == 'resnext50':
        # model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        model = resnext50_32x4d()
        model.load_state_dict(torch.load(osp.join(weights_path, 'resnext50_32x4d-1a0047aa.pth')))
        num_ftrs = model.fc.in_features
        linear_cl = torch.nn.Linear(num_ftrs, n_classes)
        if dropout > 0:
            model.fc = nn.Sequential(nn.Dropout(p=dropout), linear_cl)
        else: model.fc = linear_cl        
        
    elif model_name == 'mobilenet_v2':
        model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V2)
        num_ftrs = model.classifier[-1].in_features
        linear_cl = nn.Linear(in_features=num_ftrs, out_features=n_classes)
        if dropout > 0:
            model.classifier = nn.Sequential(nn.Dropout(p=dropout), linear_cl)
        else: model.classifier = linear_cl

    elif model_name == 'convnext':

        model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        num_ftrs = model.classifier[-1].in_features
        if dropout > 0:
            model.classifier = nn.Sequential(LayerNorm2d([num_ftrs, ], eps=1e-06, elementwise_affine=True),
                                             nn.Flatten(start_dim=1, end_dim=-1), nn.Dropout(p=dropout),
                                             nn.Linear(in_features=768, out_features=n_classes))
        else:
            model.classifier = nn.Sequential(LayerNorm2d([num_ftrs, ], eps=1e-06, elementwise_affine=True),
                                         nn.Flatten(start_dim=1, end_dim=-1),
                                         nn.Linear(in_features=768, out_features=n_classes))

    elif model_name == 'swin_t':
        model = swin_t()
        model.load_state_dict(torch.load(osp.join(weights_path, 'swin_t-704ceda3.pth')))
        num_ftrs = model.head.in_features
        linear_cl = nn.Linear(in_features=num_ftrs, out_features=n_classes)
        if dropout > 0:
            model.head = nn.Sequential(nn.Dropout(p=dropout), linear_cl)
        else: model.head = linear_cl
    else:
        sys.exit('{} is not a valid model_name, check utils.model_factory_mh.py'.format(model_name))

    setattr(model, 'n_heads', n_heads)
    setattr(model, 'n_classes', n_classes)
    return model



