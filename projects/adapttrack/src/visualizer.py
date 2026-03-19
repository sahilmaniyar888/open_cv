"""Visualization utilities for tracking results."""

from __future__ import annotations

import cv2
import numpy as np

# color palette for track IDs
COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
    (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128),
    (255, 128, 0), (128, 255, 0), (0, 128, 255), (255, 0, 128),
]


def get_color(track_id: int) -> tuple[int, int, int]:
    return COLORS[track_id % len(COLORS)]


def draw_tracks(frame: np.ndarray, tracks: list[dict],
                show_uncertainty: bool = True,
                show_trail: bool = False) -> np.ndarray:
    """Draw tracked bounding boxes with IDs and uncertainty indicators."""
    vis = frame.copy()

    for t in tracks:
        bbox = t["bbox"]
        tid = t["track_id"]
        score = t.get("score", 0)
        uncertainty = t.get("uncertainty", 0)
        color = get_color(tid)

        x1, y1, x2, y2 = [int(v) for v in bbox]

        # thickness based on uncertainty (thinner = more uncertain)
        thickness = 3 if uncertainty < 0.3 else 2 if uncertainty < 0.6 else 1

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)

        # label
        label = f"ID:{tid}"
        if show_uncertainty:
            label += f" u:{uncertainty:.2f}"

        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(vis, (x1, y1 - label_size[1] - 6), (x1 + label_size[0], y1), color, -1)
        cv2.putText(vis, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # uncertainty bar
        if show_uncertainty:
            bar_w = x2 - x1
            bar_h = 4
            filled = int(bar_w * uncertainty)
            bar_color = (0, 255, 0) if uncertainty < 0.3 else (0, 165, 255) if uncertainty < 0.6 else (0, 0, 255)
            cv2.rectangle(vis, (x1, y2 + 2), (x1 + bar_w, y2 + 2 + bar_h), (100, 100, 100), -1)
            cv2.rectangle(vis, (x1, y2 + 2), (x1 + filled, y2 + 2 + bar_h), bar_color, -1)

    return vis


def draw_stats_overlay(frame: np.ndarray, stats: dict, fps: float = 0) -> np.ndarray:
    """Draw tracking statistics overlay on frame."""
    vis = frame.copy()
    h, w = vis.shape[:2]

    # semi-transparent background
    overlay = vis.copy()
    cv2.rectangle(overlay, (w - 220, 0), (w, 120), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, vis, 0.4, 0, vis)

    y = 20
    lines = [
        f"FPS: {fps:.1f}",
        f"Active: {stats.get('active_tracks', 0)}",
        f"Lost: {stats.get('lost_tracks', 0)}",
        f"Total: {stats.get('total_tracks_created', 0)}",
        f"Frame: {stats.get('frame_count', 0)}",
    ]

    for line in lines:
        cv2.putText(vis, line, (w - 210, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y += 20

    return vis
