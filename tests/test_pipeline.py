"""
test_pipeline.py — Tests for the Pipeline Orchestrator (integration tests)
"""

import cv2
import numpy as np
import pytest
from backend.cv_engine.pipeline import DocumentScanPipeline
from backend.models import EnhancementMode


def _create_document_image():
    """
    Create a synthetic document image:
    White quadrilateral on dark background with text-like lines.
    """
    bg = np.full((600, 800, 3), 40, dtype=np.uint8)

    # Draw a skewed document
    doc_points = np.array([
        [150, 80], [650, 50], [700, 500], [100, 520]
    ], dtype=np.int32)
    cv2.fillPoly(bg, [doc_points], (240, 240, 240))

    # Add text-like lines
    for i in range(5):
        y = 150 + i * 60
        cv2.line(bg, (200, y), (600, y), (60, 60, 60), 2)

    return bg


def _image_to_bytes(image):
    """Convert numpy image to JPEG bytes."""
    _, buffer = cv2.imencode(".jpg", image)
    return buffer.tobytes()


class TestDocumentScanPipeline:
    """Integration tests for the full document scanning pipeline."""

    def setup_method(self):
        """Create a fresh pipeline for each test."""
        self.pipeline = DocumentScanPipeline()

    def test_successful_scan(self):
        """Pipeline should successfully process a clear document image."""
        image = _create_document_image()
        image_bytes = _image_to_bytes(image)

        result = self.pipeline.process(image_bytes, EnhancementMode.ENHANCED)

        assert "steps" in result
        assert "result" in result
        assert "metrics" in result

    def test_all_steps_present(self):
        """All 6 intermediate steps should be in the result."""
        image = _create_document_image()
        image_bytes = _image_to_bytes(image)

        result = self.pipeline.process(image_bytes, EnhancementMode.COLOR)
        steps = result["steps"]

        expected_steps = ["original", "preprocessed", "edges", "boundary", "warped", "enhanced"]
        for step in expected_steps:
            assert step in steps, f"Missing step: {step}"

    def test_steps_are_base64_strings(self):
        """All step images should be base64-encoded strings."""
        image = _create_document_image()
        image_bytes = _image_to_bytes(image)

        result = self.pipeline.process(image_bytes, EnhancementMode.GRAYSCALE)

        for step_name, step_value in result["steps"].items():
            assert isinstance(step_value, str), f"Step '{step_name}' should be a string"
            assert len(step_value) > 100, f"Step '{step_name}' seems too short to be an image"

    def test_metrics_have_correct_fields(self):
        """Metrics should contain all expected fields."""
        image = _create_document_image()
        image_bytes = _image_to_bytes(image)

        result = self.pipeline.process(image_bytes, EnhancementMode.ENHANCED)
        metrics = result["metrics"]

        assert "sharpness_input" in metrics
        assert "sharpness_output" in metrics
        assert "contrast_input" in metrics
        assert "contrast_output" in metrics

    def test_all_modes_work(self):
        """All enhancement modes should process successfully."""
        image = _create_document_image()
        image_bytes = _image_to_bytes(image)

        for mode in EnhancementMode:
            result = self.pipeline.process(image_bytes, mode)
            assert result["result"] is not None, f"Mode {mode.value} failed"

    def test_no_document_raises_error_when_fallback_disabled(self):
        """An image without a document should raise ValueError when fallback is disabled."""
        # Create an image with no clear document boundary
        no_doc = np.random.randint(100, 150, (600, 800, 3), dtype=np.uint8)
        image_bytes = _image_to_bytes(no_doc)

        with pytest.raises(ValueError, match="No document boundary"):
            self.pipeline.process(image_bytes, EnhancementMode.ENHANCED, fallback_to_full_frame=False)

    def test_no_document_falls_back_to_full_frame(self):
        """An image without a distinct document boundary should fall back to full image frame."""
        no_doc = np.random.randint(100, 150, (600, 800, 3), dtype=np.uint8)
        image_bytes = _image_to_bytes(no_doc)

        result = self.pipeline.process(image_bytes, EnhancementMode.ENHANCED, fallback_to_full_frame=True)
        assert result["fallback_used"] is True
        assert "steps" in result
        assert "result" in result
        assert "metrics" in result

    def test_invalid_image_bytes_raises_error(self):
        """Invalid image bytes should raise ValueError."""
        with pytest.raises(ValueError):
            self.pipeline.process(b"not an image", EnhancementMode.ENHANCED)
