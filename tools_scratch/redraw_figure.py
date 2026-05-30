#!/usr/bin/env python3
"""
Redraw the geometry figure from dataset coordinates.
"""
import matplotlib.pyplot as plt
import numpy as np

# Coordinates from dataset
points = {
    "R": [100.0, 97.0], "S": [106.0, 0.0], "T": [178.0, 39.0], "U": [195.0, 121.0],
    "V": [150.0, 60.0], "W": [141.0, 186.0], "X": [61.0, 188.0], "Y": [7.0, 128.0],
    "Z": [74.0, 157.0]
}

# Lines from dataset
lines = [
    ("R", "T"), ("R", "X"), ("U", "S"), ("V", "R"), ("V", "S"),
    ("V", "T"), ("V", "U"), ("Y", "W"), ("Y", "Z"), ("Z", "R"),
    ("Z", "W"), ("Z", "X")
]

# Plot setup
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_aspect('equal')
ax.axis('off')

# Draw Lines
for line in lines:
    x_values = [points[line[0]][0], points[line[1]][0]]
    y_values = [points[line[0]][1], points[line[1]][1]]
    ax.plot(x_values, y_values, 'k-')  # 'k-' for black line

# Draw Circle (center R, radius = distance from R to S)
radius = np.linalg.norm(np.array(points["R"]) - np.array(points["S"]))
circle = plt.Circle((points["R"][0], points["R"][1]), radius, color='k', fill=False)
ax.add_artist(circle)

# Plot points
for point, coord in points.items():
    ax.plot(coord[0], coord[1], 'ko', markersize=5)  # 'ko' for black dot
    ax.text(coord[0]+2, coord[1]+2, point, fontsize=20, color='red', va='bottom', ha='left')

# Adjust plot limits
ax.set_xlim(-20, 220)
ax.set_ylim(-20, 220)

# Invert y-axis to match image coordinates (0,0 at top-left)
ax.invert_yaxis()

plt.tight_layout()
plt.savefig("/home/hw0171/reason-vlm-sft/tools_scratch/redrawn_figure.png", dpi=100, bbox_inches='tight')
print("Saved to: /home/hw0171/reason-vlm-sft/tools_scratch/redrawn_figure.png")
plt.show()
