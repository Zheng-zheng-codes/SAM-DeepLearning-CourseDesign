import torch.nn as nn
from torchvision.models import resnet18


def get_resnet18(num_classes=10):
    """
    Build a ResNet18 model adapted for CIFAR-10.

    CIFAR-10 images are 32x32, so the original ImageNet-style
    7x7 convolution and max pooling are replaced.
    """
    model = resnet18(weights=None)

    # Adapt ResNet18 for small CIFAR-10 images
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