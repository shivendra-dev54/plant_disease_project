import torch.nn as nn
import torchvision.models as models

def create_model(num_classes: int, freeze_features: bool = True):
    model = models.resnet50(pretrained=True)

    if freeze_features:
        for param in model.parameters():
            param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
