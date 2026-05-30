#!/usr/bin/env python3
"""Create a triangle with a point - is the point inside or outside?"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, ax = plt.subplots(figsize=(6, 6))
ax.set_aspect('equal')
ax.axis('off')

# Triangle vertices
A = (100, 500)
B = (500, 500)
C = (300, 150)

# Test point (inside the triangle)
P = (280, 380)

# Draw triangle
triangle = plt.Polygon([A, B, C], fill=False, edgecolor='black', linewidth=2)
ax.add_patch(triangle)

# Draw point P
ax.plot(P[0], P[1], 'ro', markersize=12)

# Labels
ax.text(A[0]-25, A[1]+15, 'A', fontsize=20, fontweight='bold')
ax.text(B[0]+10, B[1]+15, 'B', fontsize=20, fontweight='bold')
ax.text(C[0]-10, C[1]-25, 'C', fontsize=20, fontweight='bold')
ax.text(P[0]+15, P[1], 'P', fontsize=20, fontweight='bold', color='red')

ax.set_xlim(0, 600)
ax.set_ylim(0, 600)
ax.invert_yaxis()

plt.tight_layout()
plt.savefig("/home/hw0171/reason-vlm-sft/point_in_triangle.png", dpi=100,
            bbox_inches='tight', facecolor='white')
print("Saved: /home/hw0171/reason-vlm-sft/point_in_triangle.png")
print(f"\nPoint P is at: ({int(P[0]*1000/600)}, {int(P[1]*1000/600)}) in 0-1000 scale")
print("Answer: P is INSIDE the triangle")
