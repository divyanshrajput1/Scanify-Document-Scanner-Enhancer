"""
main.py — FastAPI Application Entry Point for DocuScan

Creates and configures the FastAPI application with:
  - CORS middleware (allows React frontend to communicate)
  - API router registration
  - Logging configuration
"""

import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.router import router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Create FastAPI application
app = FastAPI(
    title="DocuScan API",
    description="Document Scanner & Enhancer — A Computer Vision pipeline "
                "that detects documents in photographs, corrects perspective "
                "distortion, and enhances readability.",
    version="1.0.0"
)

# CORS middleware — allows the React frontend (typically on port 5173)
# to make requests to the FastAPI backend (typically on port 8000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API routes
app.include_router(router)

logger.info("DocuScan API initialized successfully.")
