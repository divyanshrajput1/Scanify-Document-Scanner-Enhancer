"""
test_preprocessor.py — Tests for the Image Preprocessing Module
"""

import cv2
import numpy as np
import pytest
from backend.cv_engine.preprocessor import preprocess


def _create_test_image(width=800, height=600, channels=3):
    """Create a simple test image."""
    return np.random.randint(0, 255, (height, width, channels), dtype=np.uint8)


class TestPreprocess:
    """Tests for the preprocess function."""

    def test_returns_expected_keys(self):
        """preprocess should return a dict with resized, gray, blurred, scale_factor."""
        image = _create_test_image()
        result = preprocess(image)

        assert "resized" in result
        assert "gray" in result
        assert "blurred" in result
        assert "scale_factor" in result

    def test_grayscale_is_single_channel(self):
        """Gray output should be a 2D array (single channel)."""
        image = _create_test_image()
        result = preprocess(image)

        assert len(result["gray"].shape) == 2

    def test_blurred_is_single_channel(self):
        """Blurred output should also be single channel."""
        image = _create_test_image()
        result = preprocess(image)

        assert len(result["blurred"].shape) == 2

    def test_resizing_large_image(self):
        """Images wider than PROCESSING_WIDTH should be downsized."""
        image = _create_test_image(width=2000, height=1500)
        result = preprocess(image)

        assert result["resized"].shape[1] <= 1000

    def test_small_image_not_upscaled(self):
        """Images smaller than PROCESSING_WIDTH should not be upscaled."""
        image = _create_test_image(width=500, height=400)
        result = preprocess(image)

        assert result["resized"].shape[1] == 500

    def test_scale_factor_correct(self):
        """Scale factor should correctly map resized → original dimensions."""
        image = _create_test_image(width=2000, height=1500)
        result = preprocess(image)

        expected_scale = 2000 / result["resized"].shape[1]
        assert abs(result["scale_factor"] - expected_scale) < 0.01

    def test_blurred_is_smoother_than_gray(self):
        """Blurred image should have lower high-frequency content (lower Laplacian variance)."""
        image = _create_test_image()
        result = preprocess(image)

        gray_var = cv2.Laplacian(result["gray"], cv2.CV_64F).var()
        blur_var = cv2.Laplacian(result["blurred"], cv2.CV_64F).var()

        assert blur_var < gray_var
