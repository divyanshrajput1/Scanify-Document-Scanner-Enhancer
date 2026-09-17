/**
 * MetricsDisplay — Quality metrics visualization component.
 *
 * Displays the sharpness and contrast metrics comparing
 * input and output images, with improvement indicators.
 */
export default function MetricsDisplay({ metrics }) {
  if (!metrics) return null;

  return (
    <div className="metrics-section">
      <h2>📊 Quality Metrics</h2>
      <div className="metrics-grid">
        <div className="metric-card">
          <h3>Sharpness</h3>
          <div className="metric-values">
            <div className="metric-row">
              <span className="metric-label">Input:</span>
              <span className="metric-value">{metrics.sharpness_input.toFixed(1)}</span>
            </div>
            <div className="metric-row">
              <span className="metric-label">Output:</span>
              <span className="metric-value highlight">{metrics.sharpness_output.toFixed(1)}</span>
            </div>
          </div>
          <p className="metric-improvement">{metrics.sharpness_improvement}</p>
          <p className="metric-explain">Measured by Laplacian variance — higher means sharper edges</p>
        </div>

        <div className="metric-card">
          <h3>Contrast</h3>
          <div className="metric-values">
            <div className="metric-row">
              <span className="metric-label">Input:</span>
              <span className="metric-value">{metrics.contrast_input.toFixed(1)}</span>
            </div>
            <div className="metric-row">
              <span className="metric-label">Output:</span>
              <span className="metric-value highlight">{metrics.contrast_output.toFixed(1)}</span>
            </div>
          </div>
          <p className="metric-improvement">{metrics.contrast_improvement}</p>
          <p className="metric-explain">Measured by intensity standard deviation — higher means more contrast</p>
        </div>
      </div>
    </div>
  );
}
