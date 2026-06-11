import torch.nn as nn
from torchvision.models import resnet18, resnet34


def adapt_resnet_for_cifar(model, num_classes=10):
    """
    Adapt ImageNet-style ResNet to CIFAR-10.

    CIFAR-10 images are 32x32, so the original ImageNet-style
    7x7 convolution and max pooling are not suitable.

    Changes:
        1. Replace 7x7 conv with 3x3 conv.
        2. Set stride from 2 to 1.
        3. Remove the max pooling layer.
        4. Replace the final classifier for CIFAR-10.
    """
    model.conv1 = nn.Conv2d(
        in_channels=3,
        out_channels=64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False
    )

    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model


def get_resnet18(num_classes=10):
    """
    Build a ResNet18 model adapted for CIFAR-10.

    This model is used as the shallower residual network
    in the depth comparison experiment.
    """
    model = resnet18(weights=None)
    model = adapt_resnet_for_cifar(model, num_classes)
    return model


def get_resnet34(num_classes=10):
    """
    Build a ResNet34 model adapted for CIFAR-10.

    This model is used as the deeper residual network
    in the depth comparison experiment.
    """
    model = resnet34(weights=None)
    model = adapt_resnet_for_cifar(model, num_classes)
    return model