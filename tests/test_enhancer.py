"""
test_enhancer.py — Tests for the Image Enhancement Module
"""

import cv2
import numpy as np
import pytest
from backend.cv_engine.enhancer import (
    enhance_color, enhance_grayscale, enhance_bw, enhance_readability, enhance
)
from backend.models import EnhancementMode


def _create_test_image(width=300, height=200):
    """Create a low-contrast test image (simulating a poorly lit document)."""
    # Create image with narrow intensity range (low contrast)
    image = np.random.randint(100, 160, (height, width, 3), dtype=np.uint8)
    return image


class TestEnhanceColor:
    """Tests for the color enhancement mode."""

    def test_output_is_color(self):
        """Color enhancement should return a 3-channel image."""
        image = _create_test_image()
        result = enhance_color(image)

        assert len(result.shape) == 3
        assert result.shape[2] == 3

    def test_output_shape_matches_input(self):
        """Output dimensions should match input."""
        image = _create_test_image()
        result = enhance_color(image)

        assert result.shape == image.shape


class TestEnhanceGrayscale:
    """Tests for the grayscale enhancement mode."""

    def test_output_is_three_channel(self):
        """Should return 3-channel (BGR) for API consistency."""
        image = _create_test_image()
        result = enhance_grayscale(image)

        assert len(result.shape) == 3
        assert result.shape[2] == 3

    def test_improves_contrast(self):
        """CLAHE should increase the standard deviation of intensities."""
        image = _create_test_image()
        result = enhance_grayscale(image)

        input_std = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).std()
        output_std = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY).std()

        # CLAHE should increase contrast (std dev) on a low-contrast input
        assert output_std > input_std


class TestEnhanceBW:
    """Tests for the black & white enhancement mode."""

    def test_output_is_binary(self):
        """B&W output should contain only 0 and 255 values."""
        image = _create_test_image()
        result = enhance_bw(image)
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

        unique_values = set(np.unique(gray))
        assert unique_values.issubset({0, 255})


class TestEnhanceReadability:
    """Tests for the enhanced readability mode."""

    def test_output_is_three_channel(self):
        """Should return 3-channel image."""
        image = _create_test_image()
        result = enhance_readability(image)

        assert len(result.shape) == 3
        assert result.shape[2] == 3


class TestEnhanceDispatcher:
    """Tests for the enhance() dispatcher function."""

    def test_all_modes_produce_output(self):
        """All enhancement modes should produce a valid output."""
        image = _create_test_image()

        for mode in EnhancementMode:
            result = enhance(image, mode)
            assert result is not None
            assert result.shape[0] > 0
            assert result.shape[1] > 0

    def test_invalid_mode_raises_error(self):
        """An invalid mode should raise ValueError."""
        image = _create_test_image()

        with pytest.raises(ValueError):
            enhance(image, "nonexistent_mode")
