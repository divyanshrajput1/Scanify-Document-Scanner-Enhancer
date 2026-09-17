"""
router.py — API Route Definitions for DocuScan

Defines three endpoints:
  GET  /api/health      — Health check
  POST /api/scan        — Scan a single document image
  POST /api/scan/batch  — Scan multiple document images

Each endpoint includes input validation, error handling, and
structured JSON responses.
"""

import logging
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from backend.models import EnhancementMode, MAX_IMAGE_SIZE_MB, MAX_BATCH_SIZE, SUPPORTED_FORMATS
from backend.schemas import ScanResult, BatchScanResult, HealthResponse, QualityMetrics
from backend.cv_engine.pipeline import DocumentScanPipeline

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api")

# Initialize the CV pipeline once
pipeline = DocumentScanPipeline()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint to verify the API is running."""
    return HealthResponse(status="healthy", version="1.0.0")


def _validate_image_file(file: UploadFile) -> None:
    """
    Validate that the uploaded file is a supported image format and size.

    Raises:
        HTTPException: If the file is invalid.
    """
    # Check content type
    if file.content_type not in SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format: {file.content_type}. "
                   f"Supported formats: JPEG, PNG."
        )

    # Check filename
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="File must have a filename."
        )


async def _read_and_validate_file(file: UploadFile) -> bytes:
    """
    Read file content and validate size.

    Returns:
        bytes: The file content.

    Raises:
        HTTPException: If the file is too large.
    """
    content = await file.read()
    size_mb = len(content) / (1024 * 1024)

    if size_mb > MAX_IMAGE_SIZE_MB:
        raise HTTPException(
            status_code=400,
            detail=f"File too large: {size_mb:.1f} MB. Maximum: {MAX_IMAGE_SIZE_MB} MB."
        )

    if len(content) == 0:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty."
        )

    return content


@router.post("/scan", response_model=ScanResult)
async def scan_document(
    file: UploadFile = File(..., description="Document image to scan"),
    mode: str = Form(default="enhanced", description="Enhancement mode: color, grayscale, bw, enhanced"),
    fallback_to_full_frame: bool = Form(default=True, description="Fallback to full frame if no document contour is detected")
):
    """
    Scan a single document image.

    Processes the image through the full CV pipeline:
    1. Preprocess (resize, blur, grayscale conversion)
    2. Edge detection (Canny)
    3. Document boundary detection (contour finding)
    4. Perspective correction (homography warp)
    5. Enhancement (based on selected mode)
    6. Quality metrics computation

    Returns intermediate step images, the final scan, and quality metrics.
    """
    # Validate enhancement mode
    try:
        enhancement_mode = EnhancementMode(mode.lower())
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid enhancement mode: '{mode}'. "
                   f"Valid modes: {[m.value for m in EnhancementMode]}"
        )

    # Validate file
    _validate_image_file(file)
    content = await _read_and_validate_file(file)

    # Process through the CV pipeline
    try:
        result = pipeline.process(content, enhancement_mode, fallback_to_full_frame=fallback_to_full_frame)
    except ValueError as e:
        # Known processing errors (e.g., no document found)
        return ScanResult(
            success=False,
            message=str(e),
            mode=enhancement_mode.value
        )
    except Exception as e:
        logger.error(f"Pipeline processing error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal processing error: {str(e)}"
        )

    message = (
        "Document scanned successfully (full image frame used as boundary)."
        if result.get("fallback_used")
        else "Document scanned successfully."
    )

    return ScanResult(
        success=True,
        message=message,
        steps=result["steps"],
        result=result["result"],
        metrics=QualityMetrics(**result["metrics"]),
        mode=enhancement_mode.value
    )


@router.post("/scan/batch", response_model=BatchScanResult)
async def scan_batch(
    files: list[UploadFile] = File(..., description="Document images to scan (max 10)"),
    mode: str = Form(default="enhanced", description="Enhancement mode"),
    fallback_to_full_frame: bool = Form(default=True, description="Fallback to full frame if no document contour is detected")
):
    """
    Scan multiple document images in a single request.

    Each image is processed independently. Individual failures
    do not affect other images in the batch.
    """
    # Validate batch size
    if len(files) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Too many files: {len(files)}. Maximum: {MAX_BATCH_SIZE}."
        )

    if len(files) == 0:
        raise HTTPException(
            status_code=400,
            detail="No files provided."
        )

    # Validate enhancement mode
    try:
        enhancement_mode = EnhancementMode(mode.lower())
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid enhancement mode: '{mode}'. "
                   f"Valid modes: {[m.value for m in EnhancementMode]}"
        )

    results = []
    success_count = 0
    fail_count = 0

    for file in files:
        try:
            _validate_image_file(file)
            content = await _read_and_validate_file(file)
            result = pipeline.process(content, enhancement_mode, fallback_to_full_frame=fallback_to_full_frame)

            msg = (
                f"'{file.filename}' scanned successfully (full image boundary used)."
                if result.get("fallback_used")
                else f"'{file.filename}' scanned successfully."
            )

            results.append(ScanResult(
                success=True,
                message=msg,
                steps=result["steps"],
                result=result["result"],
                metrics=QualityMetrics(**result["metrics"]),
                mode=enhancement_mode.value
            ))
            success_count += 1

        except HTTPException as e:
            results.append(ScanResult(
                success=False,
                message=f"'{file.filename}': {e.detail}",
                mode=enhancement_mode.value
            ))
            fail_count += 1

        except ValueError as e:
            results.append(ScanResult(
                success=False,
                message=f"'{file.filename}': {str(e)}",
                mode=enhancement_mode.value
            ))
            fail_count += 1

        except Exception as e:
            logger.error(f"Batch processing error for '{file.filename}': {e}", exc_info=True)
            results.append(ScanResult(
                success=False,
                message=f"'{file.filename}': Internal processing error.",
                mode=enhancement_mode.value
            ))
            fail_count += 1

    return BatchScanResult(
        results=results,
        summary={
            "success": success_count,
            "failed": fail_count,
            "total": len(files)
        }
    )
