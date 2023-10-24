import os.path
import re
from torchvision import models as models
import torch
from torch import nn
import timm


def get_model(name_model, num_classes, requires_grad=False, model_weights=None, pretrained=True):
    if re.compile("swin*").match(name_model) is not None:
        model = timm.create_model(name_model, pretrained=pretrained)
        num_features = model.head.in_features
        for param in model.parameters():
            param.requires_grad = requires_grad

        model.head = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes))

        for param in model.head:
            param.requires_grad = True

    elif re.compile("resnet*").match(name_model) is not None:
        model = resnet_models(name_model, pretrained=pretrained, progress=True)
        num_features = model.fc.in_features

        for param in model.parameters():
            param.requires_grad = requires_grad

        model.fc = nn.Linear(num_features, num_classes)

    elif re.compile("efficientnet*").match(name_model) is not None:
        model = timm.create_model(name_model, pretrained=pretrained)
        num_features = model.classifier.in_features
        for param in model.parameters():
            param.requires_grad = requires_grad

        model.classifier = nn.Linear(num_features, num_classes)

    else:
        raise NotImplementedError()

    if model_weights is not None:
        model.load_state_dict(model_weights)

    return model


def resnet_models(name_model, pretrained=True, progress=True):
    if name_model == "resnet101":
        model = models.resnet101(progress=progress, weights=pretrained)
    elif name_model == "resnet50":
        model = models.resnet50(progress=progress, weights=pretrained)
    elif name_model == "resnet152":
        model = models.resnet152(progress=progress, weights=pretrained)
    elif name_model == "resnet18":
        model = models.resnet18(progress=progress, weights=pretrained)
    elif name_model == "resnet34":
        model = models.resnet34(progress=progress, weights=pretrained)
    else:
        raise NotImplementedError(f"A model named {name_model} was not found")
    return model
