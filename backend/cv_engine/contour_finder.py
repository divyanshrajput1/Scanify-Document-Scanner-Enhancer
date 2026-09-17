"""
contour_finder.py — Document Boundary Detection Module

CV Algorithms Used:
  - Contour Detection (Module 3 — Image Segmentation)
  - Polygon Approximation / Ramer-Douglas-Peucker (Module 3)

Purpose:
  Finds the document boundary in the edge map by:
  1. Finding all contours (connected boundaries) in the binary edge image
  2. Sorting contours by area (largest first)
  3. Approximating each contour as a polygon
  4. Selecting the largest contour that approximates to exactly 4 vertices
     (a quadrilateral = a document)

Why Polygon Approximation?
  Raw contours may have hundreds of points tracing the exact edge pixels.
  The Ramer-Douglas-Peucker algorithm simplifies a contour to fewer points
  by removing points that are within epsilon distance of the simplified line.
  For a document, this should reduce the contour to exactly 4 corner points.
"""

import cv2
import numpy as np
from typing import Optional


def find_document_contour(edge_map: np.ndarray, min_area_ratio: float = 0.05) -> Optional[np.ndarray]:
    """
    Find the document boundary in the edge map.

    Strategy:
      1. Find all contours in the edge image
      2. Filter by minimum area (must be at least min_area_ratio of image area)
      3. Approximate each contour to a polygon
      4. Return the largest contour that approximates to 4 vertices

    Args:
        edge_map: Binary edge image from Canny.
        min_area_ratio: Minimum contour area as a fraction of image area.
                        Default 0.05 means the document must occupy at least
                        5% of the image. This filters out small noise contours.

    Returns:
        4-point contour (numpy array of shape (4, 1, 2)) if found, else None.
    """
    image_height, image_width = edge_map.shape[:2]
    image_area = image_height * image_width
    min_area = image_area * min_area_ratio

    # Find all contours
    # RETR_LIST retrieves all contours without hierarchy (simplest mode)
    # CHAIN_APPROX_SIMPLE compresses horizontal/vertical/diagonal segments to endpoints
    contours, _ = cv2.findContours(edge_map, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    # Sort by area (largest first) — the document should be the biggest quadrilateral
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for contour in contours:
        area = cv2.contourArea(contour)

        # Skip contours that are too small to be a document
        if area < min_area:
            break  # Since sorted by area, all remaining are smaller

        # Approximate the contour to a polygon
        # Test multiple epsilon values (2% to 5% of perimeter) to robustly
        # detect quadrilaterals even with minor edge irregularities or curved corners
        perimeter = cv2.arcLength(contour, closed=True)
        for eps_factor in [0.02, 0.03, 0.04, 0.05]:
            approx = cv2.approxPolyDP(contour, eps_factor * perimeter, closed=True)
            # A document boundary is a convex quadrilateral (4 vertices)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                return approx

    return None


def contour_to_points(contour: np.ndarray) -> np.ndarray:
    """
    Convert a contour array to a simple (N, 2) array of (x, y) points.

    OpenCV contours have shape (N, 1, 2). This reshapes to (N, 2)
    for easier manipulation.

    Args:
        contour: Contour from findContours, shape (N, 1, 2).

    Returns:
        Points array of shape (N, 2).
    """
    return contour.reshape(-1, 2)
