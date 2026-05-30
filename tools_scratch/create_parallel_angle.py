#!/usr/bin/env python3
"""
Geometry problem: Find angle APB where A and B are on parallel lines.

Setup:
- Two parallel horizontal lines
- Point A on top line, Point B on bottom line
- Point P between the lines
- Given angles that PA and PB make with the horizontal
- Find: angle APB

Solution requires drawing a line through P parallel to the other lines,
then using alternate interior angles.
"""
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(8, 6))
ax.set_aspect('equal')
ax.axis('off')

# Two parallel horizontal lines
y_top = 100
y_bottom = 500
x_left, x_right = 50, 550

ax.plot([x_left, x_right], [y_top, y_top], 'k-', linewidth=2)
ax.plot([x_left, x_right], [y_bottom, y_bottom], 'k-', linewidth=2)

# Point P in the middle
P = (200, 300)

# Calculate A and B positions based on desired angles
# Both A and B are to the RIGHT of P
angle1 = 35 * np.pi / 180  # 35 degrees
angle2 = 45 * np.pi / 180  # 45 degrees

# A is on top line (y = y_top), to the RIGHT of P
# From P, going up-right at angle 35° above horizontal
dy_to_top = P[1] - y_top  # distance up to top line
dx_to_A = dy_to_top / np.tan(angle1)
A = (P[0] + dx_to_A, y_top)

# B is on bottom line (y = y_bottom), to the RIGHT of P
# From P, going down-right at angle 45° below horizontal
dy_to_bottom = y_bottom - P[1]  # distance down to bottom line
dx_to_B = dy_to_bottom / np.tan(angle2)
B = (P[0] + dx_to_B, y_bottom)

print(f"A = {A}")
print(f"P = {P}")
print(f"B = {B}")

# Draw segments PA and PB
ax.plot([P[0], A[0]], [P[1], A[1]], 'b-', linewidth=2)
ax.plot([P[0], B[0]], [P[1], B[1]], 'b-', linewidth=2)

# Mark points
for point, name, offset in [(A, 'A', (-15, -25)), (B, 'B', (10, 10)), (P, 'P', (-30, -5))]:
    ax.plot(*point, 'ko', markersize=8)
    ax.text(point[0]+offset[0], point[1]+offset[1], name, fontsize=18, fontweight='bold')

# Add angle arcs and labels
from matplotlib.patches import Arc

# No angle labels on the image - they will be given in the prompt

# Labels for parallel lines
ax.text(x_right + 10, y_top, 'ℓ₁', fontsize=14)
ax.text(x_right + 10, y_bottom, 'ℓ₂', fontsize=14)

ax.set_xlim(0, 620)
ax.set_ylim(0, 600)
ax.invert_yaxis()

plt.tight_layout()
plt.savefig("/home/hw0171/reason-vlm-sft/parallel_angle.png", dpi=100,
            bbox_inches='tight', facecolor='white')
print("Saved: /home/hw0171/reason-vlm-sft/parallel_angle.png")

print("""
Problem: Lines ℓ₁ and ℓ₂ are parallel.
         Point A is on ℓ₁, point B is on ℓ₂, point P is between them.
         The angle between PA and ℓ₁ is 35°.
         The angle between PB and ℓ₂ is 45°.
         Find angle APB.

Solution: Draw line through P parallel to ℓ₁ and ℓ₂.
- PA goes UP from P, PB goes DOWN from P
- Angles are on OPPOSITE sides of the auxiliary line
- angle APB = 35° + 45° = 80°

Answer: 80°
""")

print("\nCoordinates in 0-1000 scale:")
print(f"  A: ({int(A[0]*1000/620)}, {int(A[1]*1000/600)})")
print(f"  P: ({int(P[0]*1000/620)}, {int(P[1]*1000/600)})")
print(f"  B: ({int(B[0]*1000/620)}, {int(B[1]*1000/600)})")
