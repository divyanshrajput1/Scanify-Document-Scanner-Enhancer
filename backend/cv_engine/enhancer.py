"""
enhancer.py — Image Enhancement Module

CV Algorithms Used:
  - Histogram Equalization (Module 1 — Histogram Processing)
  - CLAHE — Contrast Limited Adaptive Histogram Equalization (Module 1)
  - Adaptive Thresholding (Module 1 — Histogram Processing)
  - Unsharp Masking (Module 1 — Convolution & Filtering, Image Enhancement)

Purpose:
  Applies the selected enhancement mode to the perspective-corrected
  document to improve readability.

Enhancement Modes:
  1. COLOR: Enhances color image using CLAHE on the L channel (LAB space)
  2. GRAYSCALE: Converts to grayscale, applies CLAHE for contrast
  3. BW: Converts to grayscale, applies adaptive thresholding for clean B&W
  4. ENHANCED: Grayscale + CLAHE + unsharp masking for maximum readability

Why CLAHE instead of standard Histogram Equalization?
  Standard HE redistributes the entire image's histogram uniformly,
  which can over-amplify noise in flat regions and wash out details.
  CLAHE divides the image into tiles (e.g., 8×8), equalizes each tile
  independently (with a clip limit to prevent over-amplification),
  and blends tile boundaries with bilinear interpolation. This handles
  uneven illumination (common in document photos) much better.
"""

import cv2
import numpy as np
from backend.models import EnhancementMode


def enhance_color(image: np.ndarray) -> np.ndarray:
    """
    Enhance a color document scan.

    Applies CLAHE to the Lightness channel in LAB color space.
    LAB separates luminance (L) from color (A, B), so we can
    enhance contrast without distorting colors.

    Args:
        image: Perspective-corrected BGR image.

    Returns:
        Color-enhanced BGR image.
    """
    # Convert BGR → LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    # Apply CLAHE to the L (lightness) channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l_channel)

    # Merge channels and convert back to BGR
    lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
    result = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    return result


def enhance_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Enhance a grayscale document scan using CLAHE.

    Args:
        image: Perspective-corrected BGR image.

    Returns:
        Grayscale CLAHE-enhanced image (single channel, converted to BGR for consistency).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Convert back to 3-channel for consistent API response format
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)


def enhance_bw(image: np.ndarray) -> np.ndarray:
    """
    Create a clean black-and-white scan using adaptive thresholding.

    How Adaptive Thresholding Works:
      For each pixel, the threshold is computed as the Gaussian-weighted
      mean of a local neighborhood (block_size × block_size) minus a
      constant C. Pixels above the threshold become white (255);
      below become black (0).

      This handles uneven illumination because the threshold varies
      across the image based on local brightness.

    Args:
        image: Perspective-corrected BGR image.

    Returns:
        Binary (black and white) image as BGR.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply adaptive Gaussian thresholding
    # block_size=11: neighborhood size (must be odd)
    # C=7: constant subtracted from mean (controls sensitivity)
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=11,
        C=7
    )

    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


def enhance_readability(image: np.ndarray) -> np.ndarray:
    """
    Maximum readability enhancement: CLAHE + Unsharp Masking.

    Unsharp Masking works by:
      1. Blurring the image (Gaussian, sigma~1)
      2. Subtracting the blur from the original to isolate high-frequency details
      3. Adding the details back with amplification: result = original + α * (original - blur)

    This sharpens text edges while CLAHE improves overall contrast.

    Args:
        image: Perspective-corrected BGR image.

    Returns:
        Enhanced grayscale image (as BGR) with improved readability.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 1: CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Step 2: Unsharp masking for sharpening
    # Create the blurred version
    blurred = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=1.0)

    # Sharpened = original + strength * (original - blurred)
    # Using cv2.addWeighted: result = α * src1 + β * src2 + γ
    # where α = 1 + strength, β = -strength, so result = original + strength * (original - blurred)
    sharpening_strength = 1.5
    sharpened = cv2.addWeighted(
        enhanced, 1.0 + sharpening_strength,
        blurred, -sharpening_strength,
        0
    )

    return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)


def enhance(image: np.ndarray, mode: EnhancementMode) -> np.ndarray:
    """
    Apply the selected enhancement mode to a perspective-corrected image.

    This is the main entry point for the enhancement module.

    Args:
        image: Perspective-corrected BGR image.
        mode: Enhancement mode (COLOR, GRAYSCALE, BW, ENHANCED).

    Returns:
        Enhanced image in BGR format.

    Raises:
        ValueError: If an invalid mode is provided.
    """
    enhancers = {
        EnhancementMode.COLOR: enhance_color,
        EnhancementMode.GRAYSCALE: enhance_grayscale,
        EnhancementMode.BW: enhance_bw,
        EnhancementMode.ENHANCED: enhance_readability,
    }

    enhancer_func = enhancers.get(mode)
    if enhancer_func is None:
        raise ValueError(f"Unknown enhancement mode: {mode}")

    return enhancer_func(image)
