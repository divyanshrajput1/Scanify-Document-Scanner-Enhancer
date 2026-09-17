#Scanify — Document Scanner & Enhancer

A web-based document scanning and enhancement application that transforms photographs of documents into clean, readable, perspective-corrected digital scans using classical Computer Vision algorithms.

Built as a university project for the Computer Vision course (VIT).

## Features

- **Automatic Document Detection** — Detects document boundaries using Canny edge detection and contour analysis
- **Perspective Correction** — Corrects skew and perspective distortion using homography-based warping
- **4 Enhancement Modes** — Color, Grayscale, Black & White, and Enhanced Readability
- **Pipeline Visualization** — View each intermediate processing step (edges, boundary, warp, enhancement)
- **Quality Metrics** — Sharpness and contrast scores comparing input vs. output
- **Batch Processing** — Scan up to 10 documents in a single request
- **Download** — Download scanned output as JPEG

## Technologies

| Layer | Technology |
|-------|-----------|
| Frontend | React (Vite) |
| Backend | Python, FastAPI |
| Computer Vision | OpenCV, NumPy |
| Testing | pytest, FastAPI TestClient |

## Computer Vision Algorithms Used

| Algorithm | Syllabus Module | Purpose |
|-----------|----------------|---------|
| Gaussian Blur | M1 — Convolution & Filtering | Noise reduction before edge detection |
| Canny Edge Detection | M3 — Feature Extraction | Detecting document edges |
| Morphological Operations | M1 — Filtering | Closing gaps in edge map |
| Contour Detection | M3 — Image Segmentation | Finding document boundary |
| Polygon Approximation | M3 — Segmentation | Simplifying contour to 4 corners |
| Homography / Perspective Warp | M2 — Homography | Correcting perspective distortion |
| CLAHE | M1 — Histogram Processing | Adaptive contrast enhancement |
| Adaptive Thresholding | M1 — Histogram Processing | Binary (B&W) document scan |
| Unsharp Masking | M1 — Image Enhancement | Text sharpening |

## System Architecture

```
React Frontend (port 5173)
        ↓ HTTP
FastAPI Backend (port 8000)
        ↓
CV Engine (Python + OpenCV)
  ├── Preprocessor (resize, blur, grayscale)
  ├── Edge Detector (Canny + morphology)
  ├── Contour Finder (document boundary)
  ├── Perspective Corrector (homography warp)
  ├── Enhancer (4 modes)
  └── Quality Metrics (sharpness, contrast)
```

## Installation

### Prerequisites

- Python 3.10+ ([download](https://www.python.org/downloads/))
- Node.js 18+ ([download](https://nodejs.org/))
- Git

### Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd CV-Project
   ```

2. **Set up the Python backend:**
   ```bash
   python -m venv venv

   # Windows
   .\venv\Scripts\activate

   # macOS/Linux
   source venv/bin/activate

   pip install -r requirements.txt
   ```

3. **Set up the React frontend:**
   ```bash
   cd frontend
   npm install
   cd ..
   ```

## Running the Application

You need **two terminal windows** — one for the backend, one for the frontend.

### Terminal 1 — Backend (FastAPI)
```bash
# From the project root directory
.\venv\Scripts\activate
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```
The API will be available at: http://localhost:8000  
API docs (Swagger): http://localhost:8000/docs

### Terminal 2 — Frontend (React)
```bash
# From the project root directory
cd frontend
npm run dev
```
The web app will be available at: http://localhost:5173

### Usage

1. Open http://localhost:5173 in your browser
2. Upload a document photo (JPEG or PNG)
3. Select an enhancement mode
4. Click **Scan Document**
5. View the pipeline steps, final scan, and quality metrics
6. Download the scanned output

## Testing

Run all tests from the project root:

```bash
.\venv\Scripts\activate
python -m pytest tests/ -v
```

Expected output: **58 tests passed**.

### Test Coverage

| Test File | Module Tested | Tests |
|-----------|--------------|-------|
| `test_preprocessor.py` | Image preprocessing | 7 |
| `test_edge_detector.py` | Canny edge detection | 5 |
| `test_contour_finder.py` | Document boundary detection | 5 |
| `test_perspective.py` | Homography & perspective warp | 8 |
| `test_enhancer.py` | Enhancement modes | 8 |
| `test_quality_metrics.py` | Quality scoring | 9 |
| `test_pipeline.py` | Full pipeline integration | 7 |
| `test_api.py` | API endpoints | 9 |

## Project Structure

```
CV Project/
├── backend/
│   ├── main.py                 # FastAPI entry point
│   ├── router.py               # API routes
│   ├── schemas.py              # Pydantic models
│   ├── models.py               # Enums & constants
│   └── cv_engine/
│       ├── pipeline.py         # Pipeline orchestrator
│       ├── preprocessor.py     # Resize, blur, grayscale
│       ├── edge_detector.py    # Canny + morphology
│       ├── contour_finder.py   # Document boundary
│       ├── perspective.py      # Homography warp
│       ├── enhancer.py         # 4 enhancement modes
│       ├── quality_metrics.py  # Sharpness & contrast
│       └── utils.py            # Image format conversion
├── frontend/
│   └── src/
│       ├── App.jsx             # Main component
│       ├── App.css             # Styles
│       ├── components/         # UI components
│       └── services/api.js     # API client
├── tests/                      # 58 unit & integration tests
├── sample_images/              # Test document photos
├── requirements.txt            # Python dependencies
├── statement.md                # Problem statement
└── README.md                   # This file
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Health check |
| POST | `/api/scan` | Scan a single document |
| POST | `/api/scan/batch` | Scan multiple documents |

## Limitations

1. Requires a contrasting background — fails if document color matches the surface
2. Assumes the document is the largest quadrilateral in the frame
3. Cannot handle severely crumpled or folded documents
4. Does not perform OCR — only produces a clean image scan
5. Requires both backend and frontend servers to be running

## Future Enhancements

- OCR integration (Tesseract) for text extraction
- Multi-document detection in a single image
- PDF generation from batch scans
- Real-time camera preview with live boundary detection

## License

University project — not licensed for commercial use.
