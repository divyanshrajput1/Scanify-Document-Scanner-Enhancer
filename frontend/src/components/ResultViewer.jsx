/**
 * ResultViewer — Final scan result display with download functionality.
 *
 * Shows the final enhanced scan and provides a download button.
 */
export default function ResultViewer({ result, mode }) {
  if (!result) return null;

  const handleDownload = () => {
    // Convert base64 to blob and trigger download
    const byteCharacters = atob(result);
    const byteNumbers = new Array(byteCharacters.length);
    for (let i = 0; i < byteCharacters.length; i++) {
      byteNumbers[i] = byteCharacters.charCodeAt(i);
    }
    const byteArray = new Uint8Array(byteNumbers);
    const blob = new Blob([byteArray], { type: "image/jpeg" });

    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `docuscan_${mode}_${Date.now()}.jpg`;
    link.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="result-section">
      <h2>📋 Scanned Result</h2>
      <div className="result-container">
        <img
          src={`data:image/jpeg;base64,${result}`}
          alt="Scanned document"
          className="result-image"
        />
      </div>
      <button className="download-btn" onClick={handleDownload}>
        ⬇️ Download Scan
      </button>
    </div>
  );
}
