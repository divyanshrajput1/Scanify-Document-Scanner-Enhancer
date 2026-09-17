"""
perspective.py — Perspective Correction Module

CV Algorithms Used:
  - Homography / Perspective Transform (Module 2 — Homography)
  - Projective Transformation (Module 1 — Projective Transformations)

Purpose:
  Transforms the detected document from its skewed perspective in the
  photo into a flat, top-down rectangular view — the core "scanning" step.

How Homography Works:
  A homography is a 3×3 matrix H that maps points from one plane to
  another in projective (homogeneous) coordinates:

      [x']     [h11 h12 h13] [x]
      [y']  =  [h21 h22 h23] [y]
      [w']     [h31 h32 h33] [1]

  where (x'/w', y'/w') are the transformed coordinates.

  Given 4 source points (detected corners) and 4 destination points
  (target rectangle), the 8 unknowns (h33 is normalized to 1) can be
  solved with exactly 4 point correspondences.

  cv2.getPerspectiveTransform() solves this system.
  cv2.warpPerspective() applies the transformation to every pixel.

Why This is Module 2 (Multi-Camera Views):
  Homography is the fundamental transformation in multi-view geometry.
  When a camera views a planar surface (like a document), the
  relationship between the world plane and the image plane is a
  homography. Correcting perspective is equivalent to finding the
  homography that "un-does" the camera's viewpoint.
"""

import cv2
import numpy as np


def order_points(points: np.ndarray) -> np.ndarray:
    """
    Order 4 corner points in a consistent order:
    [top-left, top-right, bottom-right, bottom-left]

    This is critical because getPerspectiveTransform requires points
    to be in a known order to produce the correct mapping.

    Algorithm:
      - Top-left has the smallest sum (x + y)
      - Bottom-right has the largest sum (x + y)
      - Top-right has the smallest difference (y - x)
      - Bottom-left has the largest difference (y - x)

    Args:
        points: Unordered array of 4 points, shape (4, 2).

    Returns:
        Ordered array of shape (4, 2): [TL, TR, BR, BL].
    """
    ordered = np.zeros((4, 2), dtype=np.float32)

    # Sum of coordinates: top-left has min sum, bottom-right has max sum
    s = points.sum(axis=1)
    ordered[0] = points[np.argmin(s)]  # Top-left
    ordered[2] = points[np.argmax(s)]  # Bottom-right

    # Difference of coordinates: top-right has min diff, bottom-left has max diff
    d = np.diff(points, axis=1).flatten()
    ordered[1] = points[np.argmin(d)]  # Top-right
    ordered[3] = points[np.argmax(d)]  # Bottom-left

    return ordered


def compute_output_dimensions(ordered_points: np.ndarray) -> tuple[int, int]:
    """
    Compute the output image dimensions from the ordered corner points.

    Uses the maximum of the top/bottom edge widths and left/right edge
    heights to determine the output rectangle size. This preserves the
    document's aspect ratio as much as possible.

    Args:
        ordered_points: Ordered corners [TL, TR, BR, BL], shape (4, 2).

    Returns:
        Tuple of (width, height) for the output image.
    """
    tl, tr, br, bl = ordered_points

    # Width: max of top edge and bottom edge
    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    width = int(max(width_top, width_bottom))

    # Height: max of left edge and right edge
    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)
    height = int(max(height_left, height_right))

    return width, height


def correct_perspective(image: np.ndarray, contour_points: np.ndarray) -> np.ndarray:
    """
    Apply perspective correction to extract a flat, top-down view of the document.

    Steps:
      1. Order the 4 detected corner points consistently
      2. Compute the output rectangle dimensions
      3. Define the destination points (a rectangle)
      4. Compute the homography matrix (3×3 perspective transform)
      5. Warp the image using the homography

    Args:
        image: Original input image (full resolution).
        contour_points: 4 corner points of the detected document, shape (4, 2).

    Returns:
        Perspective-corrected (warped) image of the document.
    """
    # Step 1: Order points consistently
    ordered = order_points(contour_points.astype(np.float32))

    # Step 2: Compute output dimensions from the quadrilateral
    width, height = compute_output_dimensions(ordered)

    # Ensure minimum dimensions
    width = max(width, 100)
    height = max(height, 100)

    # Step 3: Define destination points (a perfect rectangle)
    destination = np.array([
        [0, 0],              # Top-left
        [width - 1, 0],      # Top-right
        [width - 1, height - 1],  # Bottom-right
        [0, height - 1]      # Bottom-left
    ], dtype=np.float32)

    # Step 4: Compute the 3×3 homography matrix
    # This solves: destination = H × source (in homogeneous coordinates)
    H = cv2.getPerspectiveTransform(ordered, destination)

    # Step 5: Apply the perspective warp
    warped = cv2.warpPerspective(image, H, (width, height))

    return warped
