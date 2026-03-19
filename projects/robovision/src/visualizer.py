"""
visualizer.py — Visualization of Detection + Depth Results
===========================================================

WHAT YOU'LL LEARN HERE:
- Drawing bounding boxes and text with OpenCV
- Creating depth map color visualizations (colormaps)
- Compositing multiple visualizations into a single output
- Saving publication-quality result images

KEY CONCEPT: Good visualization is essential for debugging CV pipelines.
If you can't see what your model is doing, you can't fix it.
"""

import cv2
import numpy as np
import os
from typing import List

from src.detector import Detection
from src.scene_graph import SceneGraph


# Color palette for different depth zones
ZONE_COLORS = {
    "near": (0, 255, 0),     # Green — within reach
    "mid":  (0, 165, 255),   # Orange — visible
    "far":  (0, 0, 255),     # Red — background
    "unknown": (200, 200, 200),
}

# Color for manipulable vs non-manipulable objects
MANIP_COLOR = (0, 255, 128)    # Bright green
NON_MANIP_COLOR = (255, 128, 0)  # Blue-ish


def draw_detections(
    image: np.ndarray,
    detections: List[Detection],
) -> np.ndarray:
    """
    Draw bounding boxes and labels on the image.

    Args:
        image: RGB image (H, W, 3), uint8
        detections: List of Detection objects

    Returns:
        Image with drawn detections (RGB, uint8)
    """
    # Work on a copy to avoid modifying the original
    canvas = image.copy()

    for det in detections:
        x1, y1, x2, y2 = [int(b) for b in det.bbox]

        # Choose color based on whether object is manipulable
        color = MANIP_COLOR if det.is_manipulable else NON_MANIP_COLOR

        # Draw bounding box (thickness=2 for visibility)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

        # Prepare label text
        label = f"{det.class_name} {det.confidence:.0%}"

        # Draw label background (filled rectangle for readability)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        (text_w, text_h), baseline = cv2.getTextSize(
            label, font, font_scale, thickness
        )
        cv2.rectangle(
            canvas,
            (x1, y1 - text_h - 8),
            (x1 + text_w + 4, y1),
            color,
            -1,  # Filled
        )

        # Draw label text
        cv2.putText(
            canvas, label,
            (x1 + 2, y1 - 4),
            font, font_scale,
            (0, 0, 0),  # Black text on colored background
            thickness,
        )

    return canvas


def colorize_depth_map(depth_map: np.ndarray) -> np.ndarray:
    """
    Convert a grayscale depth map to a colorized visualization.

    Uses the INFERNO colormap: purple (near) → yellow (far).
    This is perceptually uniform — equal depth differences look
    equally different visually.

    Args:
        depth_map: (H, W) float32, values in [0, 1]

    Returns:
        Colorized depth image (H, W, 3), RGB, uint8
    """
    # Scale to 0-255 for OpenCV colormap
    depth_uint8 = (depth_map * 255).astype(np.uint8)

    # Apply colormap
    # COLORMAP_INFERNO: dark purple (near) → bright yellow (far)
    depth_colored = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_INFERNO)

    # OpenCV colormaps output BGR, convert to RGB
    depth_colored = cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)

    return depth_colored


def draw_combined_result(
    image: np.ndarray,
    scene_graph: SceneGraph,
    depth_map: np.ndarray,
) -> np.ndarray:
    """
    Create a combined visualization with detections + depth info.

    Shows bounding boxes colored by depth zone with depth values.

    Args:
        image: Original RGB image
        scene_graph: Scene graph with all object information
        depth_map: Normalized depth map

    Returns:
        Combined visualization (RGB, uint8)
    """
    canvas = image.copy()

    for obj in scene_graph.objects:
        x1, y1, x2, y2 = [int(b) for b in obj.bbox_2d]
        cx, cy = int(obj.centroid_2d[0]), int(obj.centroid_2d[1])

        # Color by depth zone
        color = ZONE_COLORS.get(obj.spatial_zone, (200, 200, 200))

        # Draw bounding box
        thickness = 3 if obj.is_manipulable else 1
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)

        # Draw centroid
        cv2.circle(canvas, (cx, cy), 4, color, -1)

        # Label with class + depth
        label = f"{obj.class_name} d={obj.estimated_depth:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX

        (tw, th), _ = cv2.getTextSize(label, font, 0.45, 1)
        cv2.rectangle(canvas, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(canvas, label, (x1 + 2, y1 - 4), font, 0.45, (0, 0, 0), 1)

        # Mark manipulable objects with a star
        if obj.is_manipulable:
            cv2.putText(canvas, "*", (x2 - 15, y1 + 15), font, 0.6, color, 2)

    # Add legend
    legend_y = 25
    for zone, zcolor in ZONE_COLORS.items():
        if zone == "unknown":
            continue
        cv2.rectangle(canvas, (10, legend_y - 12), (26, legend_y + 2), zcolor, -1)
        cv2.putText(canvas, zone, (30, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        legend_y += 20

    return canvas


def create_output_grid(
    original: np.ndarray,
    detection_vis: np.ndarray,
    depth_vis: np.ndarray,
    combined_vis: np.ndarray,
) -> np.ndarray:
    """
    Create a 2x2 grid of all visualizations.

    Layout:
      [Original Image]  [Detections]
      [Depth Map]        [Combined]

    Args:
        original: Original RGB image
        detection_vis: Image with detection boxes
        depth_vis: Colorized depth map
        combined_vis: Combined detection + depth visualization

    Returns:
        Grid image (RGB, uint8)
    """
    # Ensure all images are the same size
    h, w = original.shape[:2]
    target_size = (w, h)

    imgs = [original, detection_vis, depth_vis, combined_vis]
    resized = []
    for img in imgs:
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, target_size)
        resized.append(img)

    # Stack into 2x2 grid
    top_row = np.hstack([resized[0], resized[1]])
    bottom_row = np.hstack([resized[2], resized[3]])
    grid = np.vstack([top_row, bottom_row])

    return grid


def save_results(
    output_dir: str,
    original: np.ndarray,
    detection_vis: np.ndarray,
    depth_vis: np.ndarray,
    combined_vis: np.ndarray,
) -> None:
    """Save all visualization results to disk."""
    os.makedirs(output_dir, exist_ok=True)

    def save_rgb(image, filename):
        """Helper: convert RGB → BGR for OpenCV saving."""
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        path = os.path.join(output_dir, filename)
        cv2.imwrite(path, bgr)
        print(f"  Saved: {path}")

    print("\nSaving results:")
    save_rgb(detection_vis, "detection_result.jpg")
    save_rgb(depth_vis, "depth_map.jpg")
    save_rgb(combined_vis, "combined_result.jpg")

    # Also save the 2x2 grid
    grid = create_output_grid(original, detection_vis, depth_vis, combined_vis)
    save_rgb(grid, "result_grid.jpg")
