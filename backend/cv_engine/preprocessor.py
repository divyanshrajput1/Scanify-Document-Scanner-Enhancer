"""
preprocessor.py — Image Preprocessing Module

CV Algorithms Used:
  - Gaussian Blur (Module 1 — Convolution & Filtering)
  - Color Space Conversion (Module 1 — Image Formation)

Purpose:
  Prepares the input image for edge detection by:
  1. Resizing to a consistent processing width (for speed)
  2. Converting to grayscale (edges are intensity-based)
  3. Applying Gaussian blur to suppress noise

Why Gaussian Blur?
  Canny edge detection computes image gradients, which amplify
  high-frequency noise. Pre-blurring with a Gaussian kernel
  smooths out noise while preserving strong edges (like document
  boundaries). The Gaussian kernel is separable, making it
  computationally efficient.
"""

import cv2
import numpy as np
from backend.cv_engine.utils import resize_image
from backend.models import PROCESSING_WIDTH


def preprocess(image: np.ndarray) -> dict:
    """
    Preprocess an image for the document detection pipeline.

    Steps:
      1. Resize to PROCESSING_WIDTH (default 1000px) for consistent speed
      2. Convert BGR → Grayscale
      3. Apply Gaussian blur (5×5 kernel)

    Args:
        image: Input BGR image (numpy array).

    Returns:
        Dictionary with:
          - 'resized': Resized color image
          - 'gray': Grayscale version
          - 'blurred': Gaussian-blurred grayscale image
          - 'scale_factor': Ratio of original to resized (for mapping back)
    """
    original_height, original_width = image.shape[:2]

    # Step 1: Resize for consistent processing speed
    resized = resize_image(image, PROCESSING_WIDTH)
    resized_height, resized_width = resized.shape[:2]

    # Compute scale factor to map detected points back to original resolution
    scale_factor = original_width / resized_width

    # Step 2: Convert to grayscale
    # Edge detection works on intensity (single channel), not color
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # Step 3: Gaussian blur
    # Kernel size 5×5 provides moderate smoothing.
    # Sigma=0 lets OpenCV compute sigma from kernel size: sigma = 0.3 * ((ksize-1)*0.5 - 1) + 0.8
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    return {
        "resized": resized,
        "gray": gray,
        "blurred": blurred,
        "scale_factor": scale_factor
    }
