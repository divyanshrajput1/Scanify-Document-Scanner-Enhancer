import { useRef, useState } from "react";

/**
 * ImageUpload — File upload component with drag-and-drop support.
 *
 * Allows users to upload document images via:
 * - Click to browse
 * - Drag and drop
 *
 * Shows a preview of the selected image before scanning.
 */
export default function ImageUpload({ onFileSelect, disabled }) {
  const fileInputRef = useRef(null);
  const [preview, setPreview] = useState(null);
  const [dragActive, setDragActive] = useState(false);

  const handleFile = (file) => {
    if (!file) return;

    // Validate file type
    if (!file.type.startsWith("image/")) {
      alert("Please select an image file (JPEG or PNG).");
      return;
    }

    // Validate file size (10 MB max)
    if (file.size > 10 * 1024 * 1024) {
      alert("File is too large. Maximum size is 10 MB.");
      return;
    }

    // Create preview
    const reader = new FileReader();
    reader.onload = (e) => setPreview(e.target.result);
    reader.readAsDataURL(file);

    onFileSelect(file);
  };

  const handleInputChange = (e) => {
    handleFile(e.target.files[0]);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragActive(false);
    handleFile(e.dataTransfer.files[0]);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setDragActive(true);
  };

  const handleDragLeave = () => {
    setDragActive(false);
  };

  return (
    <div className="upload-section">
      <h2>📄 Upload Document</h2>
      <div
        className={`upload-zone ${dragActive ? "drag-active" : ""} ${disabled ? "disabled" : ""}`}
        onClick={() => !disabled && fileInputRef.current?.click()}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
      >
        {preview ? (
          <div className="preview-container">
            <img src={preview} alt="Document preview" className="preview-image" />
            <p className="preview-label">Click or drop to change image</p>
          </div>
        ) : (
          <div className="upload-placeholder">
            <span className="upload-icon">📷</span>
            <p>Click or drag & drop a document photo here</p>
            <p className="upload-hint">Supports JPEG, PNG (max 10 MB)</p>
          </div>
        )}
      </div>
      <input
        ref={fileInputRef}
        type="file"
        accept="image/jpeg,image/png"
        onChange={handleInputChange}
        style={{ display: "none" }}
        disabled={disabled}
      />
    </div>
  );
}
