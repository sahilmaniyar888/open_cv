#!/usr/bin/env python3
"""Quick script to generate sample images for visual inspection."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

# add parent to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.scene_generator import SceneGenerator
from src.domain_randomization import DomainRandomizer


def generate_sample_grid(n_cols: int = 4, n_rows: int = 3,
                         output_path: str = "samples/sample_grid.jpg"):
    """Generate a grid of sample synthetic images."""
    gen = SceneGenerator(image_size=(320, 240))
    rand = DomainRandomizer()

    cell_w, cell_h = 320, 240
    grid = np.zeros((cell_h * n_rows, cell_w * n_cols, 3), dtype=np.uint8)

    for row in range(n_rows):
        for col in range(n_cols):
            seed = row * n_cols + col
            scene = gen.generate(seed=seed)
            image = scene["image"]

            # apply randomization to half the images
            if seed % 2 == 1:
                image, _ = rand.apply(image, scene["labels"])

            # draw labels on image
            for lbl in scene["labels"]:
                x1, y1, x2, y2 = lbl["bbox"]
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 1)
                cv2.putText(image, lbl["class_name"][:4], (x1, y1 - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)

            y_off = row * cell_h
            x_off = col * cell_w
            grid[y_off:y_off + cell_h, x_off:x_off + cell_w] = image

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, grid)
    print(f"Saved sample grid to {output_path}")


if __name__ == "__main__":
    generate_sample_grid()
