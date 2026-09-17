"""
Generate a synthetic test document image for pipeline testing.
Creates a white rectangle (document) on a dark background,
rotated at an angle to simulate a perspective-distorted photo.
"""
import cv2
import numpy as np
import os

# Create a 800x600 dark background
bg = np.full((600, 800, 3), 40, dtype=np.uint8)

# Define a skewed quadrilateral (simulating a document at an angle)
doc_points = np.array([
    [150, 80],   # top-left
    [650, 50],   # top-right
    [700, 500],  # bottom-right
    [100, 520],  # bottom-left
], dtype=np.int32)

# Fill the quadrilateral with white (document)
cv2.fillPoly(bg, [doc_points], (240, 240, 240))

# Add some text-like lines inside the document
for i in range(8):
    y = 120 + i * 50
    x_start = 180 + i * 5
    x_end = 620 + i * 2
    cv2.line(bg, (x_start, y), (x_end, y), (60, 60, 60), 2)

# Add a "title" line
cv2.line(bg, (250, 100), (550, 95), (30, 30, 30), 3)

# Save
output_dir = os.path.join(os.path.dirname(__file__), "..", "sample_images")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "test_document.jpg")
cv2.imwrite(output_path, bg)
print(f"Test image saved to: {output_path}")
print(f"Image size: {bg.shape[1]}x{bg.shape[0]}")
