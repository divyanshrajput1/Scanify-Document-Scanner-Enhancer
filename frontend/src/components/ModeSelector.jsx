/**
 * ModeSelector — Enhancement mode selection component.
 *
 * Allows the user to choose between 4 enhancement modes,
 * each displayed with a description of what it does.
 */
export default function ModeSelector({ selectedMode, onModeChange, disabled }) {
  const modes = [
    {
      value: "enhanced",
      label: "✨ Enhanced Readability",
      description: "CLAHE + Sharpening for maximum legibility",
    },
    {
      value: "color",
      label: "🎨 Color Scan",
      description: "Perspective-corrected with color enhancement",
    },
    {
      value: "grayscale",
      label: "🔲 Grayscale Scan",
      description: "Grayscale with contrast enhancement",
    },
    {
      value: "bw",
      label: "⬛ Black & White",
      description: "Clean adaptive thresholding for text documents",
    },
  ];

  return (
    <div className="mode-section">
      <h2>🎛️ Enhancement Mode</h2>
      <div className="mode-grid">
        {modes.map((mode) => (
          <button
            key={mode.value}
            className={`mode-card ${selectedMode === mode.value ? "active" : ""}`}
            onClick={() => onModeChange(mode.value)}
            disabled={disabled}
          >
            <span className="mode-label">{mode.label}</span>
            <span className="mode-desc">{mode.description}</span>
          </button>
        ))}
      </div>
    </div>
  );
}
