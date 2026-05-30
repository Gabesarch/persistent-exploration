#!/usr/bin/env python3
"""
Test projecting Geometry3K point coordinates onto the image.
"""
import asyncio
import os
import sys
from pathlib import Path

os.environ["no_proxy"] = "localhost,127.0.0.1"
os.environ["NO_PROXY"] = "localhost,127.0.0.1"

sys.path.insert(0, str(Path(__file__).parent / "eval"))

from lmms_eval.mcp import MCPClient
import shutil
import tempfile

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
MCP_SERVER_PATH = str(Path(__file__).parent / "eval/lmms_eval/mcp/simplevqa_tools.py")


def pixel_to_1000(x_pixel, y_pixel):
    """Convert pixel coordinates to 0-1000 scale."""
    x_1000 = int((x_pixel / IMG_WIDTH) * 1000)
    y_1000 = int((y_pixel / IMG_HEIGHT) * 1000)
    return x_1000, y_1000


async def main():
    mcp_client = MCPClient(MCP_SERVER_PATH)
    work_dir = tempfile.mkdtemp()

    # Copy image to work dir
    image_work_path = os.path.join(work_dir, "math_example.png")
    shutil.copy(IMAGE_PATH, image_work_path)

    print(f"Work dir: {work_dir}")
    print(f"Image: {image_work_path}")
    print(f"Original image size: {IMG_WIDTH} x {IMG_HEIGHT}")
    print()

    # Convert and print all coordinates
    print("Point coordinates (pixel -> 0-1000 scale):")
    print("-" * 50)
    for name, (x_px, y_px) in POINT_POSITIONS.items():
        x_1000, y_1000 = pixel_to_1000(x_px, y_px)
        print(f"  {name}: ({x_px:6.1f}, {y_px:6.1f}) -> ({x_1000:4d}, {y_1000:4d})")
    print()

    # Draw each point on the image
    # NOTE: Each tool call returns modified image but doesn't persist it.
    # We need to save after each call.
    import base64

    print("Drawing points on image...")
    colors = ["red", "blue", "green", "orange", "purple", "cyan", "magenta", "yellow", "lime"]

    def save_result(result, path):
        """Save tool result image back to disk."""
        for c in result.content:
            if hasattr(c, 'data'):
                img_bytes = base64.b64decode(c.data)
                with open(path, "wb") as f:
                    f.write(img_bytes)
                return True
        return False

    for i, (name, (x_px, y_px)) in enumerate(POINT_POSITIONS.items()):
        x_1000, y_1000 = pixel_to_1000(x_px, y_px)
        color = colors[i % len(colors)]

        # Draw a small circle at the point
        print(f"  Drawing point {name} at ({x_1000}, {y_1000}) color={color}...")
        result = await mcp_client.run_tool("draw_circle", {
            "image_path": image_work_path,
            "x": x_1000,
            "y": y_1000,
            "radius": 20,
            "color": color,
        })
        save_result(result, image_work_path)  # Persist changes

        # Add label
        result = await mcp_client.run_tool("draw_text", {
            "image_path": image_work_path,
            "x": x_1000 + 30,
            "y": y_1000 - 10,
            "text": name,
            "color": color,
            "font_size": 24,
        })
        save_result(result, image_work_path)  # Persist changes

    # Copy final result
    output_path = os.path.join(work_dir, "annotated_with_points.png")
    shutil.copy(image_work_path, output_path)
    print(f"\nAnnotated image saved to: {output_path}")

    print(f"\nDone! Check {work_dir} for results.")


if __name__ == "__main__":
    asyncio.run(main())
