"""
test_edge_detector.py — Tests for the Edge Detection Module
"""

import cv2
import numpy as np
import pytest
from backend.cv_engine.edge_detector import detect_edges


class TestDetectEdges:
    """Tests for the detect_edges function."""

    def test_output_is_binary(self):
        """Edge map should contain only 0 and 255."""
        gray = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        edges = detect_edges(blurred)
        unique_values = set(np.unique(edges))

        assert unique_values.issubset({0, 255})

    def test_output_shape_matches_input(self):
        """Edge map should have same dimensions as input."""
        gray = np.zeros((200, 300), dtype=np.uint8)
        edges = detect_edges(gray)

        assert edges.shape == (200, 300)

    def test_uniform_image_has_no_edges(self):
        """A perfectly uniform image should produce no edges."""
        uniform = np.full((100, 100), 128, dtype=np.uint8)
        edges = detect_edges(uniform)

        assert np.sum(edges) == 0

    def test_strong_edge_is_detected(self):
        """A clear black-to-white boundary should be detected as an edge."""
        image = np.zeros((100, 200), dtype=np.uint8)
        image[:, 100:] = 255  # Right half is white
        blurred = cv2.GaussianBlur(image, (5, 5), 0)

        edges = detect_edges(blurred)

        # There should be edges near the boundary (column ~100)
        assert np.sum(edges) > 0

    def test_custom_thresholds(self):
        """Different thresholds should produce different results."""
        image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        blurred = cv2.GaussianBlur(image, (5, 5), 0)

        edges_low = detect_edges(blurred, low_threshold=10, high_threshold=50)
        edges_high = detect_edges(blurred, low_threshold=100, high_threshold=200)

        # Lower thresholds should generally detect more edges
        assert np.sum(edges_low) >= np.sum(edges_high)
