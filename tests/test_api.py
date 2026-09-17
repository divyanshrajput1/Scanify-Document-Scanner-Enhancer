"""
test_api.py — API Endpoint Tests for DocuScan

Uses httpx and FastAPI's TestClient to test endpoints
without starting a real server.
"""

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from backend.main import app


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


def _create_test_image_bytes():
    """Create a valid document image as JPEG bytes."""
    bg = np.full((600, 800, 3), 40, dtype=np.uint8)
    doc_points = np.array([
        [150, 80], [650, 50], [700, 500], [100, 520]
    ], dtype=np.int32)
    cv2.fillPoly(bg, [doc_points], (240, 240, 240))
    _, buffer = cv2.imencode(".jpg", bg)
    return buffer.tobytes()


class TestHealthEndpoint:
    """Tests for GET /api/health."""

    def test_health_returns_200(self, client):
        response = client.get("/api/health")
        assert response.status_code == 200

    def test_health_response_structure(self, client):
        response = client.get("/api/health")
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data


class TestScanEndpoint:
    """Tests for POST /api/scan."""

    def test_successful_scan(self, client):
        """Valid image should return successful scan result."""
        image_bytes = _create_test_image_bytes()
        response = client.post(
            "/api/scan",
            files={"file": ("test.jpg", image_bytes, "image/jpeg")},
            data={"mode": "enhanced"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "steps" in data
        assert "result" in data
        assert "metrics" in data

    def test_invalid_file_type(self, client):
        """Non-image file should return 400."""
        response = client.post(
            "/api/scan",
            files={"file": ("test.txt", b"hello world", "text/plain")},
            data={"mode": "enhanced"}
        )
        assert response.status_code == 400

    def test_invalid_mode(self, client):
        """Invalid enhancement mode should return 400."""
        image_bytes = _create_test_image_bytes()
        response = client.post(
            "/api/scan",
            files={"file": ("test.jpg", image_bytes, "image/jpeg")},
            data={"mode": "nonexistent"}
        )
        assert response.status_code == 400

    def test_empty_file(self, client):
        """Empty file should return 400."""
        response = client.post(
            "/api/scan",
            files={"file": ("test.jpg", b"", "image/jpeg")},
            data={"mode": "enhanced"}
        )
        assert response.status_code == 400

    def test_all_modes(self, client):
        """All 4 enhancement modes should work."""
        image_bytes = _create_test_image_bytes()
        for mode in ["color", "grayscale", "bw", "enhanced"]:
            response = client.post(
                "/api/scan",
                files={"file": ("test.jpg", image_bytes, "image/jpeg")},
                data={"mode": mode}
            )
            assert response.status_code == 200
            assert response.json()["success"] is True


class TestBatchScanEndpoint:
    """Tests for POST /api/scan/batch."""

    def test_batch_scan(self, client):
        """Batch of valid images should return results for each."""
        image_bytes = _create_test_image_bytes()
        response = client.post(
            "/api/scan/batch",
            files=[
                ("files", ("doc1.jpg", image_bytes, "image/jpeg")),
                ("files", ("doc2.jpg", image_bytes, "image/jpeg")),
            ],
            data={"mode": "enhanced"}
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) == 2
        assert data["summary"]["total"] == 2
        assert data["summary"]["success"] == 2

    def test_batch_too_many_files(self, client):
        """More than 10 files should return 400."""
        image_bytes = _create_test_image_bytes()
        files = [("files", (f"doc{i}.jpg", image_bytes, "image/jpeg")) for i in range(11)]
        response = client.post(
            "/api/scan/batch",
            files=files,
            data={"mode": "enhanced"}
        )
        assert response.status_code == 400
