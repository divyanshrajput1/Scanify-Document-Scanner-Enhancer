"""
pipeline.py — Document Scanning Pipeline Orchestrator

This module coordinates the entire CV pipeline:

  Input Image (bytes)
      │
      ▼
  1. Decode bytes → numpy array             (utils)
  2. Preprocess (resize, grayscale, blur)    (preprocessor)
  3. Detect edges (Canny + morphology)       (edge_detector)
  4. Find document contour (quadrilateral)   (contour_finder)
  5. Correct perspective (homography warp)   (perspective)
  6. Enhance (based on selected mode)        (enhancer)
  7. Compute quality metrics                 (quality_metrics)
      │
      ▼
  Output: dict with intermediate step images (base64),
          final result (base64), and quality metrics

Each step produces an intermediate image that is returned to the
frontend for educational visualization.
"""

import logging
import numpy as np
from backend.models import EnhancementMode
from backend.cv_engine.utils import bytes_to_image, image_to_base64, draw_contour_overlay
from backend.cv_engine.preprocessor import preprocess
from backend.cv_engine.edge_detector import detect_edges
from backend.cv_engine.contour_finder import find_document_contour, contour_to_points
from backend.cv_engine.perspective import correct_perspective
from backend.cv_engine.enhancer import enhance
from backend.cv_engine.quality_metrics import compute_metrics
import cv2

logger = logging.getLogger(__name__)


class DocumentScanPipeline:
    """
    Orchestrates the full document scanning CV pipeline.

    This class is stateless — each call to process() is independent.
    It is instantiated once in the router and reused for all requests.
    """

    def process(
        self,
        image_bytes: bytes,
        mode: EnhancementMode,
        fallback_to_full_frame: bool = True
    ) -> dict:
        """
        Process a document image through the full CV pipeline.

        Args:
            image_bytes: Raw image file content (JPEG/PNG bytes).
            mode: Enhancement mode to apply.
            fallback_to_full_frame: If True, uses the full image boundary when
                                   no distinct 4-point contour is found (e.g.
                                   for cropped screenshots or full-frame photos).

        Returns:
            Dictionary with:
              - 'steps': dict of stage_name → base64 image string
              - 'result': base64 string of the final enhanced scan
              - 'metrics': dict of quality metric values
              - 'fallback_used': boolean indicating if full-frame fallback was used

        Raises:
            ValueError: If no document boundary is detected and fallback_to_full_frame is False.
        """
        logger.info(f"Starting pipeline processing (mode: {mode.value})")
        steps = {}

        # Step 1: Decode image
        original = bytes_to_image(image_bytes)
        steps["original"] = image_to_base64(original)
        logger.info(f"Image decoded: {original.shape[1]}x{original.shape[0]}")

        # Step 2: Preprocess
        prep = preprocess(original)
        resized = prep["resized"]
        blurred = prep["blurred"]
        scale_factor = prep["scale_factor"]

        # Convert grayscale to BGR for consistent display in frontend
        steps["preprocessed"] = image_to_base64(
            cv2.cvtColor(prep["gray"], cv2.COLOR_GRAY2BGR)
        )

        # Step 3: Edge detection
        edges = detect_edges(blurred)
        steps["edges"] = image_to_base64(
            cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        )

        # Step 4: Find document contour
        contour = find_document_contour(edges)
        fallback_used = False

        if contour is None:
            if fallback_to_full_frame:
                logger.info("No distinct document boundary detected. Falling back to full image frame.")
                fallback_used = True
                h, w = resized.shape[:2]
                contour = np.array([
                    [[0, 0]],
                    [[w - 1, 0]],
                    [[w - 1, h - 1]],
                    [[0, h - 1]]
                ], dtype=np.int32)
            else:
                logger.warning("No document boundary detected in the image.")
                raise ValueError(
                    "No document boundary detected. Please ensure the document "
                    "is clearly visible against a contrasting background, and "
                    "the entire document is within the frame."
                )

        # Draw contour overlay on the resized image for visualization
        steps["boundary"] = image_to_base64(
            draw_contour_overlay(resized, contour)
        )

        # Step 5: Scale contour points back to original image resolution
        points = contour_to_points(contour)
        points_original = (points * scale_factor).astype(np.float32)

        # Step 6: Perspective correction on the original (full-resolution) image
        warped = correct_perspective(original, points_original)
        steps["warped"] = image_to_base64(warped)
        logger.info(f"Perspective corrected: {warped.shape[1]}x{warped.shape[0]}")

        # Step 7: Enhancement
        enhanced = enhance(warped, mode)
        steps["enhanced"] = image_to_base64(enhanced)

        # Step 8: Quality metrics
        metrics = compute_metrics(original, enhanced)
        logger.info(f"Quality metrics: sharpness {metrics['sharpness_input']:.1f} → {metrics['sharpness_output']:.1f}, "
                    f"contrast {metrics['contrast_input']:.1f} → {metrics['contrast_output']:.1f}")

        return {
            "steps": steps,
            "result": steps["enhanced"],  # Final result is the enhanced image
            "metrics": metrics,
            "fallback_used": fallback_used
        }
