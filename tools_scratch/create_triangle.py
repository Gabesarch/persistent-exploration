#!/usr/bin/env python3
"""Create a simple triangle image for testing drawing ability."""
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(6, 6))
ax.set_aspect('equal')
ax.axis('off')

# Triangle vertices (in pixel coords, image will be ~600x600)
A = (100, 500)  # bottom left
B = (500, 500)  # bottom right
C = (300, 100)  # top

# Draw triangle
triangle = plt.Polygon([A, B, C], fill=False, edgecolor='black', linewidth=2)
ax.add_patch(triangle)

# Label vertices
ax.text(A[0]-20, A[1]+20, 'A', fontsize=24, fontweight='bold')
ax.text(B[0]+10, B[1]+20, 'B', fontsize=24, fontweight='bold')
ax.text(C[0]-10, C[1]-20, 'C', fontsize=24, fontweight='bold')

# Set limits
ax.set_xlim(0, 600)
ax.set_ylim(0, 600)
ax.invert_yaxis()

plt.tight_layout()
plt.savefig("/home/hw0171/reason-vlm-sft/triangle_test.png", dpi=100, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print("Saved: /home/hw0171/reason-vlm-sft/triangle_test.png")

# Print coordinates in 0-1000 scale (assuming 600x600 image)
print("\nCoordinates in 0-1000 scale:")
print(f"  A: ({int(A[0]*1000/600)}, {int(A[1]*1000/600)})")
print(f"  B: ({int(B[0]*1000/600)}, {int(B[1]*1000/600)})")
print(f"  C: ({int(C[0]*1000/600)}, {int(C[1]*1000/600)})")
print(f"  Base midpoint: ({int((A[0]+B[0])/2*1000/600)}, {int(A[1]*1000/600)})")
