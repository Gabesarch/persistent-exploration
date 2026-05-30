#!/usr/bin/env python3
"""Create an image with labeled points for testing."""
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6, 6))
ax.set_aspect('equal')
ax.axis('off')
ax.set_facecolor('white')

# Points at various locations
points = {
    "P": (150, 200),
    "Q": (450, 150),
    "R": (500, 400),
    "S": (100, 450),
}

# Draw points and labels
for name, (x, y) in points.items():
    ax.plot(x, y, 'ko', markersize=10)
    ax.text(x + 15, y - 15, name, fontsize=24, fontweight='bold', color='blue')

ax.set_xlim(0, 600)
ax.set_ylim(0, 600)
ax.invert_yaxis()

plt.tight_layout()
plt.savefig("/home/hw0171/reason-vlm-sft/labeled_points.png", dpi=100,
            bbox_inches='tight', facecolor='white')
print("Saved: /home/hw0171/reason-vlm-sft/labeled_points.png")

print("\nActual coordinates (0-1000 scale):")
for name, (x, y) in points.items():
    print(f"  {name}: ({int(x*1000/600)}, {int(y*1000/600)})")
