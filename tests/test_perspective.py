"""
test_perspective.py — Tests for the Perspective Correction Module
"""

import cv2
import numpy as np
import pytest
from backend.cv_engine.perspective import order_points, compute_output_dimensions, correct_perspective


class TestOrderPoints:
    """Tests for the order_points function."""

    def test_already_ordered(self):
        """Points already in TL, TR, BR, BL order should remain unchanged."""
        points = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)
        ordered = order_points(points)

        assert np.array_equal(ordered[0], [0, 0])      # TL
        assert np.array_equal(ordered[1], [100, 0])     # TR
        assert np.array_equal(ordered[2], [100, 100])   # BR
        assert np.array_equal(ordered[3], [0, 100])     # BL

    def test_shuffled_order(self):
        """Shuffled points should be correctly reordered."""
        points = np.array([[100, 100], [0, 0], [0, 100], [100, 0]], dtype=np.float32)
        ordered = order_points(points)

        assert np.array_equal(ordered[0], [0, 0])      # TL (min sum)
        assert np.array_equal(ordered[1], [100, 0])     # TR (min diff)
        assert np.array_equal(ordered[2], [100, 100])   # BR (max sum)
        assert np.array_equal(ordered[3], [0, 100])     # BL (max diff)

    def test_output_shape(self):
        """Output should always be (4, 2) float32."""
        points = np.array([[50, 30], [200, 20], [220, 180], [40, 190]], dtype=np.float32)
        ordered = order_points(points)

        assert ordered.shape == (4, 2)
        assert ordered.dtype == np.float32


class TestComputeOutputDimensions:
    """Tests for the compute_output_dimensions function."""

    def test_square_document(self):
        """A square document should produce roughly equal width and height."""
        points = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)
        w, h = compute_output_dimensions(points)

        assert w == 100
        assert h == 100

    def test_landscape_document(self):
        """A landscape document should have width > height."""
        points = np.array([[0, 0], [200, 0], [200, 100], [0, 100]], dtype=np.float32)
        w, h = compute_output_dimensions(points)

        assert w > h

    def test_positive_dimensions(self):
        """Dimensions should always be positive."""
        points = np.array([[10, 20], [150, 15], [160, 200], [5, 210]], dtype=np.float32)
        w, h = compute_output_dimensions(points)

        assert w > 0
        assert h > 0


class TestCorrectPerspective:
    """Tests for the correct_perspective function."""

    def test_output_is_rectangular(self):
        """Output should be a valid image (non-zero dimensions)."""
        # Create a simple test image
        image = np.full((400, 600, 3), 200, dtype=np.uint8)

        # Define a document region
        points = np.array([[50, 30], [500, 20], [520, 350], [40, 360]], dtype=np.float32)

        warped = correct_perspective(image, points)

        assert warped is not None
        assert len(warped.shape) == 3  # Still a color image
        assert warped.shape[0] > 0
        assert warped.shape[1] > 0

    def test_output_contains_document_content(self):
        """The warped area should contain the document's pixel content."""
        # Create image with a colored document region
        image = np.zeros((400, 600, 3), dtype=np.uint8)
        doc_points = np.array([[50, 30], [500, 20], [520, 350], [40, 360]], dtype=np.int32)
        cv2.fillPoly(image, [doc_points], (200, 200, 200))

        points = doc_points.astype(np.float32)
        warped = correct_perspective(image, points)

        # Center of warped image should be close to document color (200, 200, 200)
        center_y, center_x = warped.shape[0] // 2, warped.shape[1] // 2
        center_pixel = warped[center_y, center_x]

        assert np.mean(center_pixel) > 150  # Should be bright (document area)
