"""
schemas.py — Pydantic Request/Response Models for DocuScan

Defines the structured response schemas for the API endpoints.
Request validation is handled via FastAPI's UploadFile and Form
parameters directly in the router, since the input is multipart
form data (not JSON).
"""

from pydantic import BaseModel
from typing import Optional


class QualityMetrics(BaseModel):
    """Quality metrics computed on the scanned output."""
    sharpness_input: float       # Laplacian variance of input image
    sharpness_output: float      # Laplacian variance of output image
    contrast_input: float        # Std dev of intensity in input
    contrast_output: float       # Std dev of intensity in output
    sharpness_improvement: str   # Human-readable improvement description
    contrast_improvement: str    # Human-readable improvement description


class ScanResult(BaseModel):
    """Result of scanning a single document image."""
    success: bool
    message: str
    steps: Optional[dict[str, str]] = None    # Stage name -> base64 image
    result: Optional[str] = None               # Final scan as base64 image
    metrics: Optional[QualityMetrics] = None
    mode: Optional[str] = None


class BatchScanResult(BaseModel):
    """Result of scanning multiple document images."""
    results: list[ScanResult]
    summary: dict[str, int]  # {"success": N, "failed": M, "total": N+M}


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
