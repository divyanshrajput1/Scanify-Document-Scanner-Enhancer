"""
models.py — Data Models & Enums for DocuScan

Defines the enhancement modes and shared data structures used
across the backend. Keeping these in a separate file avoids
circular imports and provides a single source of truth for
constant values.
"""

from enum import Enum


class EnhancementMode(str, Enum):
    """
    Available document enhancement modes.

    Each mode applies a different image processing strategy
    to the perspective-corrected document scan.
    """
    COLOR = "color"           # Color scan with basic enhancement
    GRAYSCALE = "grayscale"   # Grayscale with contrast enhancement (CLAHE)
    BW = "bw"                 # Black & white via adaptive thresholding
    ENHANCED = "enhanced"     # Maximum readability: CLAHE + sharpening


# Processing constants
MAX_IMAGE_SIZE_MB = 10
MAX_BATCH_SIZE = 10
PROCESSING_WIDTH = 1000  # Resize width for internal processing (preserves aspect ratio)
SUPPORTED_FORMATS = {"image/jpeg", "image/png", "image/jpg"}
