# Simplified visualizer with primitive drawing functions only
# Based on detectron2's Visualizer, stripped of detectron2 dependencies
import colorsys
import random

import cv2
import matplotlib as mpl
import matplotlib.colors as mplc
import matplotlib.figure as mplfigure
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg


def random_color(rgb=False, maximum=255):
    idx = random.randint(0, len(_COLORS) - 1)
    color = _COLORS[idx] * maximum
    if rgb:
        return tuple(color)
    return color


_COLORS = np.array([
    0.000, 0.447, 0.741, 0.850, 0.325, 0.098, 0.929, 0.694, 0.125,
    0.494, 0.184, 0.556, 0.466, 0.674, 0.188, 0.301, 0.745, 0.933,
    0.635, 0.078, 0.184, 0.300, 0.300, 0.300, 0.600, 0.600, 0.600,
    1.000, 0.000, 0.000, 1.000, 0.500, 0.000, 0.749, 0.749, 0.000,
    0.000, 1.000, 0.000, 0.000, 0.000, 1.000, 0.667, 0.000, 1.000,
    0.333, 0.333, 0.000, 0.333, 0.667, 0.000, 0.333, 1.000, 0.000,
    0.667, 0.333, 0.000, 0.667, 0.667, 0.000, 0.667, 1.000, 0.000,
    1.000, 0.333, 0.000, 1.000, 0.667, 0.000, 1.000, 1.000, 0.000,
    0.000, 0.333, 0.500, 0.000, 0.667, 0.500, 0.000, 1.000, 0.500,
    0.333, 0.000, 0.500, 0.333, 0.333, 0.500, 0.333, 0.667, 0.500,
    0.333, 1.000, 0.500, 0.667, 0.000, 0.500, 0.667, 0.333, 0.500,
    0.667, 0.667, 0.500, 0.667, 1.000, 0.500, 1.000, 0.000, 0.500,
    1.000, 0.333, 0.500, 1.000, 0.667, 0.500, 1.000, 1.000, 0.500,
    0.000, 0.333, 1.000, 0.000, 0.667, 1.000, 0.000, 1.000, 1.000,
    0.333, 0.000, 1.000, 0.333, 0.333, 1.000, 0.333, 0.667, 1.000,
    0.333, 1.000, 1.000, 0.667, 0.000, 1.000, 0.667, 0.333, 1.000,
    0.667, 0.667, 1.000, 0.667, 1.000, 1.000, 1.000, 0.000, 1.000,
    1.000, 0.333, 1.000, 1.000, 0.667, 1.000, 0.333, 0.000, 0.000,
    0.500, 0.000, 0.000, 0.667, 0.000, 0.000, 0.833, 0.000, 0.000,
    1.000, 0.000, 0.000, 0.000, 0.167, 0.000, 0.000, 0.333, 0.000,
    0.000, 0.500, 0.000, 0.000, 0.667, 0.000, 0.000, 0.833, 0.000,
    0.000, 1.000, 0.000, 0.000, 0.000, 0.167, 0.000, 0.000, 0.333,
    0.000, 0.000, 0.500, 0.000, 0.000, 0.667, 0.000, 0.000, 0.833,
    0.000, 0.000, 1.000, 0.000, 0.000, 0.000, 0.143, 0.143, 0.143,
    0.857, 0.857, 0.857, 1.000, 1.000, 1.000
]).astype(np.float32).reshape(-1, 3)


class VisImage:
    """Matplotlib-based image container for visualization."""

    def __init__(self, img, scale=1.0):
        self.img = img
        self.scale = scale
        self.width, self.height = img.shape[1], img.shape[0]
        self._setup_figure(img)

    def _setup_figure(self, img):
        fig = mplfigure.Figure(frameon=False)
        self.dpi = fig.get_dpi()
        fig.set_size_inches(
            (self.width * self.scale + 1e-2) / self.dpi,
            (self.height * self.scale + 1e-2) / self.dpi,
        )
        self.canvas = FigureCanvasAgg(fig)
        ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
        ax.axis("off")
        self.fig = fig
        self.ax = ax
        self.reset_image(img)

    def reset_image(self, img):
        img = img.astype("uint8")
        self.ax.imshow(img, extent=(0, self.width, self.height, 0), interpolation="nearest")

    def save(self, filepath):
        self.fig.savefig(filepath)

    def get_image(self):
        canvas = self.canvas
        s, (width, height) = canvas.print_to_buffer()
        buffer = np.frombuffer(s, dtype="uint8")
        img_rgba = buffer.reshape(height, width, 4)
        rgb = img_rgba[:, :, :3].astype("uint8")

        # Resize back to original dimensions to prevent distortion
        if rgb.shape[0] != self.height or rgb.shape[1] != self.width:
            from PIL import Image
            pil_img = Image.fromarray(rgb)
            pil_img = pil_img.resize((self.width, self.height), Image.Resampling.LANCZOS)
            rgb = np.array(pil_img)

        return rgb


class Visualizer:
    """Visualizer with primitive drawing functions."""

    def __init__(self, img_rgb, scale=1.0):
        self.img = np.asarray(img_rgb).clip(0, 255).astype(np.uint8)
        self.output = VisImage(self.img, scale=scale)
        self._default_font_size = max(
            np.sqrt(self.output.height * self.output.width) // 90, 10 // scale
        )
        self._default_font_size = 18
        self.color_proposals = [
            list(mplc.hex2color(c)) for c in mplc.CSS4_COLORS.values()
        ]

    def draw_text(self, text, position, *, font_size=None, color="g",
                  horizontal_alignment="center", rotation=0):
        """Draw text at position with auto-contrasting background."""
        if not font_size:
            font_size = self._default_font_size

        color = np.maximum(list(mplc.to_rgb(color)), 0.15)
        color[np.argmax(color)] = max(0.8, np.max(color))

        R, G, B = color * 255
        Y = 0.299 * R + 0.587 * G + 0.114 * B
        bbox_background = 'black' if Y > 128 else 'white'

        x, y = position
        self.output.ax.text(
            x, y, text,
            size=font_size * self.output.scale,
            family="sans-serif",
            bbox={"facecolor": bbox_background, "alpha": 0.8, "pad": 0.7, "edgecolor": "none"},
            verticalalignment="top",
            horizontalalignment=horizontal_alignment,
            color=color,
            zorder=10,
            rotation=rotation,
        )
        return self.output

    def draw_box(self, box_coord, alpha=0.5, edge_color="g", line_style="-", linewidth=None):
        """Draw bounding box (x0, y0, x1, y1)."""
        x0, y0, x1, y1 = box_coord
        width = x1 - x0
        height = y1 - y0
        if linewidth is None:
            linewidth = max(self._default_font_size / 12, 1)

        self.output.ax.add_patch(
            mpl.patches.Rectangle(
                (x0, y0), width, height,
                fill=False,
                edgecolor=edge_color,
                linewidth=linewidth * self.output.scale,
                alpha=alpha,
                linestyle=line_style,
            )
        )
        return self.output

    def draw_circle(self, circle_coord, color, radius=3):
        """Draw filled circle at (x, y)."""
        self.output.ax.add_patch(
            mpl.patches.Circle(circle_coord, radius=radius, fill=True, color=color)
        )
        return self.output

    def draw_line(self, x_data, y_data, color, linestyle="-", linewidth=None):
        """Draw line from x_data, y_data lists."""
        if linewidth is None:
            linewidth = self._default_font_size / 3
        linewidth = max(linewidth, 1)
        self.output.ax.add_line(
            mpl.lines.Line2D(
                x_data, y_data,
                linewidth=linewidth * self.output.scale,
                color=color,
                linestyle=linestyle,
            )
        )
        return self.output

    def draw_polygon(self, segment, color, edge_color=None, alpha=0.5):
        """Draw filled polygon from Nx2 array of points."""
        if edge_color is None:
            if alpha > 0.8:
                edge_color = self._change_color_brightness(color, brightness_factor=-0.7)
            else:
                edge_color = color
        edge_color = mplc.to_rgb(edge_color) + (1,)

        polygon = mpl.patches.Polygon(
            segment,
            fill=True,
            facecolor=mplc.to_rgb(color) + (alpha,),
            edgecolor=edge_color,
            linewidth=max(self._default_font_size // 15 * self.output.scale, 1),
        )
        self.output.ax.add_patch(polygon)
        return self.output

    def draw_binary_mask(self, binary_mask, color=None, *, edge_color=None,
                         text=None, alpha=0.7, area_threshold=10):
        """Draw binary mask with optional text label."""
        if color is None:
            color = random_color(rgb=True, maximum=1)
        color = mplc.to_rgb(color)

        binary_mask = binary_mask.astype("uint8")
        shape2d = (binary_mask.shape[0], binary_mask.shape[1])

        rgba = np.zeros(shape2d + (4,), dtype="float32")
        rgba[:, :, :3] = color
        rgba[:, :, 3] = (binary_mask == 1).astype("float32") * alpha
        self.output.ax.imshow(rgba, extent=(0, self.output.width, self.output.height, 0))

        if text is not None:
            lighter_color = self._change_color_brightness(color, brightness_factor=0.7)
            self._draw_text_in_mask(binary_mask, text, lighter_color)
        return self.output

    def draw_binary_mask_with_number(self, binary_mask, color=None, *, edge_color=None,
                                     text=None, label_mode='1', alpha=0.1,
                                     anno_mode=['Mask', 'Mark'], area_threshold=10):
        """Draw binary mask with SoM-style numbered label at center."""
        if color is None:
            randint = random.randint(0, len(self.color_proposals) - 1)
            color = self.color_proposals[randint]
        color = mplc.to_rgb(color)

        binary_mask = binary_mask.astype("uint8")
        shape2d = (binary_mask.shape[0], binary_mask.shape[1])

        if 'Mask' in anno_mode:
            rgba = np.zeros(shape2d + (4,), dtype="float32")
            rgba[:, :, :3] = color
            rgba[:, :, 3] = (binary_mask == 1).astype("float32") * alpha
            self.output.ax.imshow(rgba, extent=(0, self.output.width, self.output.height, 0))

        if 'Box' in anno_mode:
            ys, xs = np.where(binary_mask)
            if len(xs) > 0 and len(ys) > 0:
                bbox = [xs.min(), ys.min(), xs.max(), ys.max()]
                self.draw_box(bbox, edge_color=color, alpha=0.75)

        if 'Mark' in anno_mode and text is not None:
            lighter_color = [1, 1, 1]
            self._draw_number_in_mask(binary_mask, text, lighter_color, label_mode)

        return self.output

    def _draw_number_in_mask(self, binary_mask, text, color, label_mode='1'):
        """Draw number at mask center using distance transform."""
        def number_to_string(n):
            chars = []
            while n:
                n, remainder = divmod(n - 1, 26)
                chars.append(chr(97 + remainder))
            return ''.join(reversed(chars))

        binary_mask = np.pad(binary_mask, ((1, 1), (1, 1)), 'constant')
        mask_dt = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 0)
        mask_dt = mask_dt[1:-1, 1:-1]
        max_dist = np.max(mask_dt)
        if max_dist == 0:
            return
        coords_y, coords_x = np.where(mask_dt == max_dist)

        if label_mode == 'a':
            text = number_to_string(int(text))

        self.draw_text(text, (coords_x[len(coords_x) // 2] + 2, coords_y[len(coords_y) // 2] - 6), color=color)

    def _draw_text_in_mask(self, binary_mask, text, color):
        """Draw text at mask center using connected components."""
        _num_cc, cc_labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, 8)
        if stats[1:, -1].size == 0:
            return
        largest_component_id = np.argmax(stats[1:, -1]) + 1
        center = np.median((cc_labels == largest_component_id).nonzero(), axis=1)[::-1]
        self.draw_text(text, center, color=color)

    def _change_color_brightness(self, color, brightness_factor):
        """Adjust color brightness via HLS color space."""
        assert -1.0 <= brightness_factor <= 1.0
        color = mplc.to_rgb(color)
        polygon_color = colorsys.rgb_to_hls(*color)
        modified_lightness = polygon_color[1] + (brightness_factor * polygon_color[1])
        modified_lightness = max(0.0, min(1.0, modified_lightness))
        return colorsys.hls_to_rgb(polygon_color[0], modified_lightness, polygon_color[2])

    def _jitter(self, color):
        """Randomly modify color for variation."""
        color = mplc.to_rgb(color)
        vec = np.random.rand(3)
        vec = vec / np.linalg.norm(vec) * 0.5
        res = np.clip(vec + color, 0, 1)
        return tuple(res)

    def get_output(self):
        """Return the VisImage output."""
        return self.output
