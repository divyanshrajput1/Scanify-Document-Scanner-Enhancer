/**
 * api.js — API Client for DocuScan Backend
 *
 * Handles all HTTP communication with the FastAPI backend.
 * Uses the native fetch API (no additional dependencies needed).
 */

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "/api";

/**
 * Check if the backend API is healthy and reachable.
 * @returns {Promise<object>} Health status response
 */
export async function checkHealth() {
  const response = await fetch(`${API_BASE_URL}/health`);
  if (!response.ok) {
    throw new Error(`Health check failed: ${response.status}`);
  }
  return response.json();
}

/**
 * Scan a single document image.
 *
 * @param {File} file - The image file to scan
 * @param {string} mode - Enhancement mode: 'color' | 'grayscale' | 'bw' | 'enhanced'
 * @returns {Promise<object>} Scan result with steps, result, and metrics
 */
export async function scanDocument(file, mode = "enhanced") {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("mode", mode);

  const response = await fetch(`${API_BASE_URL}/scan`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({
      detail: `Server error: ${response.status}`,
    }));
    throw new Error(error.detail || "An error occurred while scanning.");
  }

  return response.json();
}

/**
 * Scan multiple document images in a batch.
 *
 * @param {File[]} files - Array of image files to scan
 * @param {string} mode - Enhancement mode
 * @returns {Promise<object>} Batch scan results
 */
export async function scanBatch(files, mode = "enhanced") {
  const formData = new FormData();
  files.forEach((file) => {
    formData.append("files", file);
  });
  formData.append("mode", mode);

  const response = await fetch(`${API_BASE_URL}/scan/batch`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({
      detail: `Server error: ${response.status}`,
    }));
    throw new Error(error.detail || "An error occurred during batch scanning.");
  }

  return response.json();
}
