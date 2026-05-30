#!/usr/bin/env python3
"""
Draw points directly with matplotlib to verify coordinate system.
"""
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

# From the dataset
IMG_WIDTH = 278
IMG_HEIGHT = 273

POINT_POSITIONS = {
    "R": [100.0, 97.0],
    "S": [106.0, 0.0],
    "T": [178.0, 39.0],
    "U": [195.0, 121.0],
    "V": [150.0, 60.0],
    "W": [141.0, 186.0],
    "X": [61.0, 188.0],
    "Y": [7.0, 128.0],
    "Z": [74.0, 157.0],
}

IMAGE_PATH = "/home/hw0171/reason-vlm-sft/math_example.png"

# Load image
img = mpimg.imread(IMAGE_PATH)
actual_height, actual_width = img.shape[:2]

print(f"Dataset says: {IMG_WIDTH} x {IMG_HEIGHT}")
print(f"Actual image: {actual_width} x {actual_height}")

# Create figure
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(img)

# Scale factors
scale_x = actual_width / IMG_WIDTH
scale_y = actual_height / IMG_HEIGHT
print(f"Scale: x={scale_x:.3f}, y={scale_y:.3f}")

colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow', 'lime']

for i, (name, (x, y)) in enumerate(POINT_POSITIONS.items()):
    # Scale to actual image coordinates
    x_scaled = x * scale_x
    y_scaled = y * scale_y

    color = colors[i % len(colors)]

    # Plot point
    ax.plot(x_scaled, y_scaled, 'o', markersize=12, color=color, markeredgecolor='white', markeredgewidth=2)

    # Add label
    ax.text(x_scaled + 10, y_scaled, name, fontsize=14, color=color, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    print(f"  {name}: ({x:.1f}, {y:.1f}) -> ({x_scaled:.1f}, {y_scaled:.1f})")

ax.set_title(f"Points from dataset (scaled {scale_x:.2f}x, {scale_y:.2f}x)")
ax.axis('off')

plt.tight_layout()
plt.savefig("/home/hw0171/reason-vlm-sft/tools_scratch/matplotlib_points.png", dpi=150, bbox_inches='tight')
print(f"\nSaved to: /home/hw0171/reason-vlm-sft/tools_scratch/matplotlib_points.png")
plt.show()
