#!/usr/bin/env python3
"""
Create two line segments that would intersect if extended.
Task: Draw the extensions to find where they meet.
"""
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6, 6))
ax.set_aspect('equal')
ax.axis('off')

# Line segment 1: from A to B (going up-right)
A = (100, 450)
B = (250, 350)

# Line segment 2: from C to D (going up-left)
C = (500, 450)
D = (350, 350)

# If extended, they meet around (300, 250)

# Draw the segments
ax.plot([A[0], B[0]], [A[1], B[1]], 'b-', linewidth=3, label='Segment 1')
ax.plot([C[0], D[0]], [C[1], D[1]], 'g-', linewidth=3, label='Segment 2')

# Label endpoints
ax.plot(*A, 'ko', markersize=8)
ax.plot(*B, 'ko', markersize=8)
ax.plot(*C, 'ko', markersize=8)
ax.plot(*D, 'ko', markersize=8)

ax.text(A[0]-20, A[1]+15, 'A', fontsize=16, fontweight='bold')
ax.text(B[0]+10, B[1]-10, 'B', fontsize=16, fontweight='bold')
ax.text(C[0]+10, C[1]+15, 'C', fontsize=16, fontweight='bold')
ax.text(D[0]-25, D[1]-10, 'D', fontsize=16, fontweight='bold')

ax.set_xlim(0, 600)
ax.set_ylim(0, 600)
ax.invert_yaxis()

plt.tight_layout()
plt.savefig("/home/hw0171/reason-vlm-sft/extend_lines.png", dpi=100,
            bbox_inches='tight', facecolor='white')
print("Saved: /home/hw0171/reason-vlm-sft/extend_lines.png")

# Calculate actual intersection
# Line 1: from (100,450) to (250,350) -> slope = (350-450)/(250-100) = -100/150 = -2/3
# y - 450 = -2/3 * (x - 100)
# Line 2: from (500,450) to (350,350) -> slope = (350-450)/(350-500) = -100/-150 = 2/3
# y - 450 = 2/3 * (x - 500)
# Solving: x = 300, y = 317 (approx)
print("\nActual intersection point (if extended): approximately (300, 317)")
print("In 0-1000 scale: (500, 528)")
