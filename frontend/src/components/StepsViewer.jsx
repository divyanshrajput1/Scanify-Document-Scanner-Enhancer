/**
 * StepsViewer — Intermediate pipeline steps visualization.
 *
 * Displays each stage of the CV pipeline as a labeled image,
 * showing the user exactly what happens at each step.
 * This is the educational/demo feature of DocuScan.
 */
import { useState } from "react";

export default function StepsViewer({ steps }) {
  const [expandedStep, setExpandedStep] = useState(null);

  if (!steps) return null;

  const stepLabels = {
    original: { label: "1. Original Input", desc: "The uploaded document photo" },
    preprocessed: { label: "2. Preprocessed", desc: "Grayscale conversion for edge detection" },
    edges: { label: "3. Edge Detection", desc: "Canny edge detection output" },
    boundary: { label: "4. Document Boundary", desc: "Detected document quadrilateral" },
    warped: { label: "5. Perspective Corrected", desc: "Homography-based perspective warp" },
    enhanced: { label: "6. Enhanced Output", desc: "Final enhanced scan" },
  };

  const stepOrder = ["original", "preprocessed", "edges", "boundary", "warped", "enhanced"];

  return (
    <div className="steps-section">
      <h2>🔬 Pipeline Steps</h2>
      <p className="steps-subtitle">
        Click any step to view it full-size. Each step shows an intermediate result from the CV pipeline.
      </p>
      <div className="steps-grid">
        {stepOrder.map((key) => {
          if (!steps[key]) return null;
          const info = stepLabels[key] || { label: key, desc: "" };
          return (
            <div
              key={key}
              className={`step-card ${expandedStep === key ? "expanded" : ""}`}
              onClick={() => setExpandedStep(expandedStep === key ? null : key)}
            >
              <img
                src={`data:image/jpeg;base64,${steps[key]}`}
                alt={info.label}
                className="step-image"
              />
              <div className="step-info">
                <strong>{info.label}</strong>
                <span>{info.desc}</span>
              </div>
            </div>
          );
        })}
      </div>

      {expandedStep && steps[expandedStep] && (
        <div className="step-expanded-overlay" onClick={() => setExpandedStep(null)}>
          <div className="step-expanded-content" onClick={(e) => e.stopPropagation()}>
            <h3>{stepLabels[expandedStep]?.label}</h3>
            <p>{stepLabels[expandedStep]?.desc}</p>
            <img
              src={`data:image/jpeg;base64,${steps[expandedStep]}`}
              alt={stepLabels[expandedStep]?.label}
              className="step-expanded-image"
            />
            <button className="close-btn" onClick={() => setExpandedStep(null)}>
              ✕ Close
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
