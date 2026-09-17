"""
test_quality_metrics.py — Tests for the Quality Metrics Module
"""

import cv2
import numpy as np
import pytest
from backend.cv_engine.quality_metrics import compute_sharpness, compute_contrast, compute_metrics


class TestComputeSharpness:
    """Tests for the sharpness scoring function."""

    def test_sharp_image_scores_higher(self):
        """A sharp image (strong edges) should score higher than a blurred one."""
        # Create image with sharp edges
        sharp = np.zeros((100, 100), dtype=np.uint8)
        sharp[40:60, 40:60] = 255

        # Blur it
        blurry = cv2.GaussianBlur(sharp, (15, 15), 5)

        assert compute_sharpness(sharp) > compute_sharpness(blurry)

    def test_uniform_image_low_sharpness(self):
        """A perfectly uniform image should have near-zero sharpness."""
        uniform = np.full((100, 100), 128, dtype=np.uint8)
        assert compute_sharpness(uniform) < 1.0

    def test_accepts_color_image(self):
        """Should work with both grayscale and color images."""
        color = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        score = compute_sharpness(color)
        assert score > 0

    def test_returns_float(self):
        """Sharpness score should be a float."""
        image = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
        assert isinstance(compute_sharpness(image), float)


class TestComputeContrast:
    """Tests for the contrast scoring function."""

    def test_high_contrast_image(self):
        """An image with both black and white pixels should have high contrast."""
        image = np.zeros((100, 100), dtype=np.uint8)
        image[:50] = 255  # Top half white

        assert compute_contrast(image) > 100  # High std dev

    def test_low_contrast_image(self):
        """A nearly uniform image should have low contrast."""
        image = np.full((100, 100), 128, dtype=np.uint8)
        assert compute_contrast(image) < 1.0

    def test_returns_float(self):
        """Contrast score should be a float."""
        image = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
        assert isinstance(compute_contrast(image), float)


class TestComputeMetrics:
    """Tests for the combined metrics function."""

    def test_returns_all_fields(self):
        """Should return all expected metric fields."""
        input_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        output_img = np.random.randint(0, 255, (80, 80, 3), dtype=np.uint8)

        metrics = compute_metrics(input_img, output_img)

        assert "sharpness_input" in metrics
        assert "sharpness_output" in metrics
        assert "contrast_input" in metrics
        assert "contrast_output" in metrics
        assert "sharpness_improvement" in metrics
        assert "contrast_improvement" in metrics

    def test_improvement_strings_are_descriptive(self):
        """Improvement descriptions should contain 'Improved' or 'Decreased'."""
        input_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        output_img = np.random.randint(0, 255, (80, 80, 3), dtype=np.uint8)

        metrics = compute_metrics(input_img, output_img)

        assert "Improved" in metrics["sharpness_improvement"] or "Decreased" in metrics["sharpness_improvement"]
