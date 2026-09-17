"""
utils.py — Shared Utility Functions for DocuScan CV Engine

Provides helper functions for:
  - Converting between image formats (bytes ↔ numpy, numpy ↔ base64)
  - Resizing images while preserving aspect ratio
  - Drawing overlays on images

These utilities are used across multiple CV engine modules to
avoid code duplication.
"""

import cv2
import numpy as np
import base64


def bytes_to_image(image_bytes: bytes) -> np.ndarray:
    """
    Convert raw bytes (from file upload) to an OpenCV image (numpy array).

    Args:
        image_bytes: Raw image file content.

    Returns:
        BGR image as numpy array.

    Raises:
        ValueError: If the bytes cannot be decoded as an image.
    """
    np_array = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Could not decode image. The file may be corrupted or not a valid image.")

    return image


def image_to_base64(image: np.ndarray, format: str = ".jpg", quality: int = 90) -> str:
    """
    Convert an OpenCV image (numpy array) to a base64-encoded string.

    This is used to send images in JSON API responses. The frontend
    can display these directly using: <img src="data:image/jpeg;base64,{string}">

    Args:
        image: BGR image as numpy array.
        format: Image format extension ('.jpg' or '.png').
        quality: JPEG quality (1-100). Ignored for PNG.

    Returns:
        Base64-encoded string of the image.
    """
    if format == ".jpg":
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    else:
        encode_params = []

    success, buffer = cv2.imencode(format, image, encode_params)

    if not success:
        raise ValueError("Failed to encode image to the specified format.")

    return base64.b64encode(buffer).decode("utf-8")


def resize_image(image: np.ndarray, target_width: int) -> np.ndarray:
    """
    Resize an image to a target width while preserving the aspect ratio.

    Used to normalize image size for consistent processing speed
    regardless of the input resolution.

    Args:
        image: Input image.
        target_width: Desired width in pixels.

    Returns:
        Resized image with preserved aspect ratio.
    """
    height, width = image.shape[:2]

    # Don't upscale small images
    if width <= target_width:
        return image.copy()

    scale = target_width / width
    new_height = int(height * scale)
    resized = cv2.resize(image, (target_width, new_height), interpolation=cv2.INTER_AREA)

    return resized


def draw_contour_overlay(image: np.ndarray, contour: np.ndarray) -> np.ndarray:
    """
    Draw the detected document contour on a copy of the image.

    Used for the intermediate visualization step to show the user
    which boundary the system detected.

    Args:
        image: Original image.
        contour: Document boundary contour (4 points).

    Returns:
        Copy of the image with the contour drawn in green.
    """
    overlay = image.copy()
    cv2.drawContours(overlay, [contour], -1, (0, 255, 0), 3)

    # Draw corner circles for visibility
    for point in contour:
        x, y = point[0] if len(point.shape) > 1 else point
        cv2.circle(overlay, (int(x), int(y)), 8, (0, 0, 255), -1)

    return overlay
