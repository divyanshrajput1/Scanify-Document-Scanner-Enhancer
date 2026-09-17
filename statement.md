# Scanify — Problem Statement

## Problem Statement

Physical documents, lecture notes, whiteboards, receipts, and handwritten notes are routinely photographed using smartphone cameras for digital archival, sharing, and reference. However, these photographs typically suffer from perspective distortion (the document appears trapezoidal rather than rectangular), uneven illumination (shadows, glare, and varying brightness), low contrast, and camera sensor noise. These issues make the captured documents difficult to read and unsuitable for professional or academic use.

Scanify addresses this problem by providing an automated, web-based document scanning and enhancement system that uses classical Computer Vision algorithms to detect document boundaries, correct perspective distortion via homography estimation, and enhance image quality for improved readability.

## Scope

**In scope:**
- Single-page document scanning from photographs
- Automatic quadrilateral detection and perspective correction
- Multiple enhancement modes (color, grayscale, black & white, enhanced readability)
- Batch processing (up to 10 images)
- Intermediate pipeline visualization for educational transparency
- Output quality metrics (sharpness, contrast)
- Web-based interface for image upload, processing, and download

**Out of scope:**
- Multi-page document handling or PDF assembly
- Optical Character Recognition (OCR) or text extraction
- Real-time camera feed processing
- Mobile application
- User authentication or cloud storage
- Handwriting recognition

## Target Users

- **University students** digitizing lecture notes, handwritten assignments, and exam answer sheets
- **Professionals** scanning receipts, business cards, whiteboard notes, or printed documents
- **General users** needing a quick, clean digital scan without access to a physical scanner

## High-Level Features

1. **Document Detection** — Automatically detects the document boundary in a photograph using Canny edge detection and contour analysis
2. **Perspective Correction** — Corrects perspective distortion to produce a flat, top-down view using homography-based warping
3. **Image Enhancement** — Enhances the scanned output with four modes: color enhancement, grayscale with CLAHE, adaptive-threshold black & white, and maximum-readability mode with sharpening
4. **Pipeline Visualization** — Displays each intermediate processing step (original, preprocessed, edges, boundary, warped, enhanced) for educational understanding
5. **Quality Metrics** — Computes and displays sharpness and contrast scores comparing input and output
6. **Batch Processing** — Processes multiple document images in a single request
