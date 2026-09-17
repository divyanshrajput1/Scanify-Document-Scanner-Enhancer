"""
edge_detector.py — Edge Detection Module

CV Algorithms Used:
  - Canny Edge Detection (Module 3 — Feature Extraction)
  - Morphological Operations (Module 1 — Convolution & Filtering)

Purpose:
  Detects edges in the preprocessed image to find document boundaries.

How Canny Works (4 stages):
  1. Gradient Computation: Uses Sobel operators to compute horizontal
     and vertical gradients. The gradient magnitude and direction
     are computed at each pixel.
  2. Non-Maximum Suppression: Thins edges to 1-pixel width by keeping
     only local maxima along the gradient direction.
  3. Double Thresholding: Classifies edge pixels as:
     - Strong (above high threshold) — definitely an edge
     - Weak (between low and high) — maybe an edge
     - Suppressed (below low) — not an edge
  4. Hysteresis: Weak edges connected to strong edges are kept;
     isolated weak edges are removed.

Why Canny for Document Detection?
  Document boundaries are strong, well-defined edges with high
  contrast against the background. Canny's hysteresis thresholding
  provides clean, connected edges ideal for contour detection.
"""

import cv2
import numpy as np


def detect_edges(blurred_gray: np.ndarray, low_threshold: int = 50, high_threshold: int = 150) -> np.ndarray:
    """
    Apply Canny edge detection followed by morphological cleanup.

    The morphological closing operation fills small gaps in detected
    edges, which helps findContours produce complete document boundaries.

    Args:
        blurred_gray: Preprocessed (blurred, grayscale) image.
        low_threshold: Canny low threshold. Edges below this are discarded.
        high_threshold: Canny high threshold. Edges above this are kept.

    Returns:
        Binary edge map (uint8, values 0 or 255).
    """
    # Step 1: Canny edge detection
    edges = cv2.Canny(blurred_gray, low_threshold, high_threshold)

    # Step 2: Morphological closing to fill small gaps in edges
    # A 5×5 rectangular kernel dilates first (fills gaps), then erodes
    # (restores edge thickness). This helps produce closed contours.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    return edges_closed
