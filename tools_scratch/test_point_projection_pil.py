#!/usr/bin/env python3
"""
Test projecting Geometry3K point coordinates onto the image using PIL directly.
"""
from PIL import Image, ImageDraw, ImageFont
import os

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


def main():
    # Load image
    img = Image.open(IMAGE_PATH).convert("RGB")
    actual_width, actual_height = img.size

    print(f"Dataset says image size: {IMG_WIDTH} x {IMG_HEIGHT}")
    print(f"Actual image size: {actual_width} x {actual_height}")
    print()

    # Use actual image dimensions for scaling
    scale_x = actual_width / IMG_WIDTH
    scale_y = actual_height / IMG_HEIGHT
    print(f"Scale factors: x={scale_x:.3f}, y={scale_y:.3f}")
    print()

    draw = ImageDraw.Draw(img)

    # Try to load a font, fall back to default
    try:
        font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", 16)
    except:
        font = ImageFont.load_default()

    colors = ["red", "blue", "green", "orange", "purple", "cyan", "magenta", "yellow", "lime"]

    print("Drawing points:")
    print("-" * 50)

    for i, (name, (x_px, y_px)) in enumerate(POINT_POSITIONS.items()):
        # Scale coordinates to actual image size
        x_actual = int(x_px * scale_x)
        y_actual = int(y_px * scale_y)

        color = colors[i % len(colors)]

        print(f"  {name}: dataset ({x_px:6.1f}, {y_px:6.1f}) -> actual ({x_actual:4d}, {y_actual:4d}) [{color}]")

        # Draw circle (ellipse with same width/height)
        r = 8
        draw.ellipse([x_actual - r, y_actual - r, x_actual + r, y_actual + r],
                     outline=color, width=2)

        # Draw label
        draw.text((x_actual + 10, y_actual - 8), name, fill=color, font=font)

    # Save result
    output_path = "/home/hw0171/reason-vlm-sft/tools_scratch/annotated_points.png"
    img.save(output_path)
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
