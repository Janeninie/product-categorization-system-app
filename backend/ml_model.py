"""Machine Learning model for product categorization."""
import json
from pathlib import Path
from typing import Optional

import safetensors.torch
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights


class _TransferModel(nn.Module):
    """Transfer learning model using a pretrained backbone."""

    def __init__(self, backbone: nn.Module, num_classes: int, freeze_backbone: bool) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.backbone = backbone

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        in_features = self._get_in_features(backbone)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes),
        )

    def _get_in_features(self, backbone: nn.Module) -> int:
        """Extract the number of input features from the backbone."""
        if hasattr(backbone, "classifier"):
            return backbone.classifier[1].in_features
        elif hasattr(backbone, "fc"):
            return backbone.fc.in_features
        elif hasattr(backbone, "head"):
            return backbone.head.in_features
        return 1280

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone.features(x) if hasattr(self.backbone, "features") else self.backbone(x)
        if hasattr(features, "flatten"):
            features = features.flatten(1)
        return self.classifier(features)


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


def _build_efficientnet_b0(num_classes: int, freeze_backbone: bool, dropout: float) -> _TransferModel:
    """Build EfficientNet-B0 model."""
    weights = EfficientNet_B0_Weights.DEFAULT
    backbone = models.efficientnet_b0(weights=weights)
    backbone.classifier = nn.Identity()
    return _TransferModel(backbone, num_classes, freeze_backbone)


_REGISTRY = {
    "efficientnet_b0": lambda nc, fb, do: _build_efficientnet_b0(nc, fb, do),
    "simple_cnn": lambda nc, fb, do: SimpleCNN(num_classes=nc, dropout=do),
}


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
    if name not in _REGISTRY:
        raise ValueError(f"Unknown model: {name}. Available: {list(_REGISTRY.keys())}")
    return _REGISTRY[name](num_classes, freeze_backbone, dropout)


def load_model_weights(model: nn.Module, weights_path: Path, device: str = "cpu") -> nn.Module:
    """Load model weights from a safetensors file.

    Args:
        model: The model to load weights into.
        weights_path: Path to the .safetensors file.
        device: Device to load weights to.

    Returns:
        The model with loaded weights.
    """
    state_dict = safetensors.torch.load_file(str(weights_path), device=device)
    model.load_state_dict(state_dict)
    return model


class ProductClassifier:
    """Wrapper for the product classification model."""

    def __init__(self, model: nn.Module, label_map: dict):
        self.model = model
        self.label_map = label_map
        self.idx_to_class = {v: k for k, v in label_map.items()}

    @classmethod
    def from_checkpoint(cls, checkpoint_path: Path, num_classes: int = 2, device: str = "cpu") -> "ProductClassifier":
        """Load a product classifier from a checkpoint."""
        checkpoint = safetensors.torch.load_file(str(checkpoint_path), device=device)

        if "num_classes" in checkpoint:
            num_classes = checkpoint["num_classes"]

        label_map = {str(i): label for i, label in enumerate(["beverage", "snack"][:num_classes])}

        model = build_model(name="efficientnet_b0", num_classes=num_classes)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

        model = model.to(device)
        model.eval()

        return cls(model, label_map)

    def predict(self, x: torch.Tensor) -> tuple[str, float]:
        """Predict the class and confidence for an input tensor.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Tuple of (predicted_class, confidence).
        """
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=-1)
            conf, preds = torch.max(probs, dim=-1)

        pred_idx = preds.item()
        confidence = conf.item()
        predicted_class = self.idx_to_class.get(pred_idx, str(pred_idx))

        return predicted_class, confidence
