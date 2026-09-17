import { useState } from "react";
import ImageUpload from "./components/ImageUpload";
import ModeSelector from "./components/ModeSelector";
import StepsViewer from "./components/StepsViewer";
import ResultViewer from "./components/ResultViewer";
import MetricsDisplay from "./components/MetricsDisplay";
import { scanDocument } from "./services/api";
import "./App.css";

/**
 * App — Main application component for DocuScan.
 *
 * Manages the application state and coordinates the workflow:
 * 1. User uploads a document photo
 * 2. User selects an enhancement mode
 * 3. User clicks "Scan" to process
 * 4. Results (steps, scan, metrics) are displayed
 */
export default function App() {
  const [file, setFile] = useState(null);
  const [mode, setMode] = useState("enhanced");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);

  const handleScan = async () => {
    if (!file) {
      setError("Please upload a document image first.");
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const data = await scanDocument(file, mode);

      if (!data.success) {
        setError(data.message || "Scanning failed. Please try a different image.");
        return;
      }

      setResult(data);
    } catch (err) {
      setError(err.message || "An error occurred. Is the backend server running?");
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setFile(null);
    setResult(null);
    setError(null);
  };

  return (
    <div className="app">
      <header className="app-header">
        <h1>📄 DocuScan</h1>
        <p className="subtitle">Document Scanner & Enhancer — Computer Vision Pipeline</p>
      </header>

      <main className="app-main">
        <div className="controls-panel">
          <ImageUpload onFileSelect={setFile} disabled={loading} />
          <ModeSelector selectedMode={mode} onModeChange={setMode} disabled={loading} />

          <div className="action-buttons">
            <button
              className="scan-btn"
              onClick={handleScan}
              disabled={!file || loading}
            >
              {loading ? "⏳ Processing..." : "🔍 Scan Document"}
            </button>

            {result && (
              <button className="reset-btn" onClick={handleReset}>
                🔄 New Scan
              </button>
            )}
          </div>

          {error && (
            <div className="error-message">
              <strong>⚠️ Error:</strong> {error}
            </div>
          )}
        </div>

        {result && (
          <div className="results-panel">
            <ResultViewer result={result.result} mode={result.mode} />
            <MetricsDisplay metrics={result.metrics} />
            <StepsViewer steps={result.steps} />
          </div>
        )}
      </main>

      <footer className="app-footer">
        <p>DocuScan — Computer Vision University Project | Built with React + FastAPI + OpenCV</p>
      </footer>
    </div>
  );
}
