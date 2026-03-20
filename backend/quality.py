"""Image quality analysis module."""
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image


@dataclass
class QualityMetrics:
    """Container for image quality metrics."""
    brightness: float
    blur_var: float
    width: int
    height: int
    quality_warnings: List[str]


def calculate_brightness(image: np.ndarray) -> float:
    """Calculate the mean pixel intensity (brightness) of an image.

    Args:
        image: RGB image as numpy array (H, W, 3).

    Returns:
        Mean pixel intensity value.
    """
    return float(np.mean(image))


def calculate_blur_var(image: np.ndarray) -> float:
    """Calculate the variance of the Laplacian to measure image blur.

    Args:
        image: RGB image as numpy array (H, W, 3).

    Returns:
        Variance of Laplacian values.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return float(laplacian.var())


def analyze_quality(pil_image: Image.Image) -> QualityMetrics:
    """Analyze the quality of an image and return metrics with warnings.

    Args:
        pil_image: PIL Image object (RGB).

    Returns:
        QualityMetrics containing brightness, blur_var, dimensions, and warnings.
    """
    width, height = pil_image.size

    img_array = np.array(pil_image)

    brightness = calculate_brightness(img_array)
    blur_var = calculate_blur_var(img_array)

    warnings = []

    if brightness < 50:
        warnings.append("low_brightness")
    elif brightness > 200:
        warnings.append("high_brightness")

    if blur_var < 30:
        warnings.append("low_blur")

    if width < 100 or height < 100:
        warnings.append("small_resolution")

    return QualityMetrics(
        brightness=brightness,
        blur_var=blur_var,
        width=width,
        height=height,
        quality_warnings=warnings,
    )
