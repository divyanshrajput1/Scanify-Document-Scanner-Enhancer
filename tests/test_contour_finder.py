"""
test_contour_finder.py — Tests for the Document Boundary Detection Module
"""

import cv2
import numpy as np
import pytest
from backend.cv_engine.contour_finder import find_document_contour, contour_to_points


def _create_document_edge_map(width=800, height=600, doc_rect=(100, 80, 650, 500)):
    """
    Create a synthetic edge map containing a rectangular document boundary.

    Args:
        doc_rect: (x1, y1, x2, y2) of the document rectangle.

    Returns:
        Binary edge map with the document boundary drawn.
    """
    edge_map = np.zeros((height, width), dtype=np.uint8)
    x1, y1, x2, y2 = doc_rect
    cv2.rectangle(edge_map, (x1, y1), (x2, y2), 255, 2)
    return edge_map


class TestFindDocumentContour:
    """Tests for the find_document_contour function."""

    def test_finds_rectangle(self):
        """Should find a clear rectangular document boundary."""
        edge_map = _create_document_edge_map()
        contour = find_document_contour(edge_map)

        assert contour is not None
        assert len(contour) == 4  # Quadrilateral

    def test_returns_none_for_empty_image(self):
        """Should return None when no contours are found."""
        edge_map = np.zeros((600, 800), dtype=np.uint8)
        contour = find_document_contour(edge_map)

        assert contour is None

    def test_ignores_small_contours(self):
        """Should ignore contours smaller than min_area_ratio."""
        edge_map = np.zeros((600, 800), dtype=np.uint8)
        # Draw a tiny rectangle (much less than 5% of image area)
        cv2.rectangle(edge_map, (10, 10), (30, 30), 255, 1)
        contour = find_document_contour(edge_map)

        assert contour is None

    def test_selects_largest_quadrilateral(self):
        """When multiple quadrilaterals exist, should select the largest."""
        edge_map = np.zeros((600, 800), dtype=np.uint8)
        # Large rectangle
        cv2.rectangle(edge_map, (50, 50), (700, 500), 255, 2)
        # Smaller rectangle
        cv2.rectangle(edge_map, (200, 200), (400, 350), 255, 2)

        contour = find_document_contour(edge_map)
        assert contour is not None

        # The detected contour should be the larger one
        area = cv2.contourArea(contour)
        large_area = (700 - 50) * (500 - 50)  # ~390000
        small_area = (400 - 200) * (350 - 200)  # ~30000

        # Should be closer to the large rectangle's area
        assert area > small_area


class TestContourToPoints:
    """Tests for the contour_to_points utility."""

    def test_reshapes_correctly(self):
        """Should reshape (N, 1, 2) contour to (N, 2)."""
        contour = np.array([[[10, 20]], [[30, 40]], [[50, 60]], [[70, 80]]])
        points = contour_to_points(contour)

        assert points.shape == (4, 2)
        assert np.array_equal(points[0], [10, 20])
