"""
quality_metrics.py — Output Quality Scoring Module

Purpose:
  Computes quantitative metrics to evaluate the quality of the
  scanned output compared to the input. These metrics are used
  in the project report's evaluation section and displayed in the UI.

Metrics:
  1. Sharpness Score (Laplacian Variance):
     The Laplacian operator computes the second derivative of the image.
     Sharp images have high-frequency edges → high Laplacian variance.
     Blurry images have smoothed edges → low Laplacian variance.

  2. Contrast Score (Intensity Standard Deviation):
     The standard deviation of pixel intensities measures how spread
     out the intensity values are. High std = high contrast (good).
     Low std = low contrast (washed out or flat).
"""

import cv2
import numpy as np


def compute_sharpness(image: np.ndarray) -> float:
    """
    Compute the sharpness score using the variance of the Laplacian.

    The Laplacian is a second-derivative operator that highlights
    regions of rapid intensity change (edges). A sharp image has
    many well-defined edges → high Laplacian variance.

    Args:
        image: Input image (BGR or grayscale).

    Returns:
        Sharpness score (higher = sharper). Typical range: 10–1000+.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Compute Laplacian (second derivative)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)

    # Variance of Laplacian = sharpness measure
    return float(laplacian.var())


def compute_contrast(image: np.ndarray) -> float:
    """
    Compute the contrast score as the standard deviation of pixel intensities.

    A well-contrasted image has a wide spread of intensity values.
    A low-contrast image has most pixels clustered near the mean.

    Args:
        image: Input image (BGR or grayscale).

    Returns:
        Contrast score (higher = more contrast). Typical range: 20–80.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    return float(gray.std())


def compute_metrics(input_image: np.ndarray, output_image: np.ndarray) -> dict:
    """
    Compute quality metrics comparing input and output images.

    Args:
        input_image: Original input image.
        output_image: Final enhanced scan.

    Returns:
        Dictionary with metric names and values, including
        human-readable improvement descriptions.
    """
    sharpness_in = compute_sharpness(input_image)
    sharpness_out = compute_sharpness(output_image)
    contrast_in = compute_contrast(input_image)
    contrast_out = compute_contrast(output_image)

    # Compute improvement descriptions
    if sharpness_in > 0:
        sharp_change = ((sharpness_out - sharpness_in) / sharpness_in) * 100
        sharp_desc = f"{'Improved' if sharp_change > 0 else 'Decreased'} by {abs(sharp_change):.1f}%"
    else:
        sharp_desc = "Input sharpness too low to compute ratio"

    if contrast_in > 0:
        contrast_change = ((contrast_out - contrast_in) / contrast_in) * 100
        contrast_desc = f"{'Improved' if contrast_change > 0 else 'Decreased'} by {abs(contrast_change):.1f}%"
    else:
        contrast_desc = "Input contrast too low to compute ratio"

    return {
        "sharpness_input": round(sharpness_in, 2),
        "sharpness_output": round(sharpness_out, 2),
        "contrast_input": round(contrast_in, 2),
        "contrast_output": round(contrast_out, 2),
        "sharpness_improvement": sharp_desc,
        "contrast_improvement": contrast_desc,
    }
