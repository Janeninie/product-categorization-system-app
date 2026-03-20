"""Machine Learning model for product categorization."""
from pathlib import Path
from typing import Optional

import safetensors.torch
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights


class ProductClassifier(nn.Module):
    """Product classifier using EfficientNet-B0 backbone.

    Architecture: EfficientNet-B0 features + AdaptiveAvgPool + Classifier head.
    """
    BACKBONE_OUT_FEATURES: int = 1280

    def __init__(
        self,
        num_classes: int = 2,
        freeze_backbone: bool = True,
        dropout: float = 0.3,
        pretrained: bool = True,
    ) -> None:
        super().__init__()

        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.efficientnet_b0(weights=weights)

        self._backbone = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(self.BACKBONE_OUT_FEATURES, num_classes),
        )

        self.num_classes = num_classes

        if freeze_backbone:
            self.freeze_backbone()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._backbone(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def freeze_backbone(self) -> None:
        for param in self._backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for param in self._backbone.parameters():
            param.requires_grad = True

    def trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "num_classes": self.num_classes,
            },
            path,
        )


class SimpleCNN(nn.Module):
    """Simple CNN model for product categorization."""

    def __init__(self, num_classes: int = 4, dropout: float = 0.3) -> None:
        super().__init__()
        self.num_classes = num_classes

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(256 * 14 * 14, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

    def trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(
    name: str = "efficientnet_b0",
    num_classes: int = 2,
    freeze_backbone: bool = False,
    dropout: float = 0.3,
) -> nn.Module:
    """Build a model by name with specified parameters.

    Args:
        name: Model name in the registry.
        num_classes: Number of output classes.
        freeze_backbone: Whether to freeze the backbone.
        dropout: Dropout rate.

    Returns:
        The built model.
    """
    if name == "efficientnet_b0":
        return ProductClassifier(
            num_classes=num_classes,
            freeze_backbone=freeze_backbone,
            dropout=dropout,
            pretrained=False,
        )
    elif name == "simple_cnn":
        return SimpleCNN(num_classes=num_classes, dropout=dropout)
    else:
        raise ValueError(f"Unknown model: {name}. Available: efficientnet_b0, simple_cnn")
