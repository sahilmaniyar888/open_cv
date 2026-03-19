"""Procedural synthetic scene generator using OpenCV and PIL."""

from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter


class SceneGenerator:
    """
    Generates synthetic training images with automatic annotations.

    Creates scenes with geometric objects, applies domain randomization,
    and exports bounding box labels in YOLO format.
    """

    OBJECT_CLASSES = {
        0: "circle",
        1: "rectangle",
        2: "triangle",
        3: "ellipse",
        4: "pentagon",
    }

    def __init__(
        self,
        image_size: tuple[int, int] = (640, 480),
        min_objects: int = 2,
        max_objects: int = 8,
        min_obj_size: int = 30,
        max_obj_size: int = 120,
    ):
        self.image_size = image_size
        self.min_objects = min_objects
        self.max_objects = max_objects
        self.min_obj_size = min_obj_size
        self.max_obj_size = max_obj_size

    def generate(self, seed: int | None = None) -> dict:
        """
        Generate a single synthetic scene.

        Returns:
            dict with keys:
                image: np.ndarray (H, W, 3) BGR
                labels: list of dicts with class_id, bbox [x1,y1,x2,y2]
                yolo_labels: list of YOLO-format strings
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        w, h = self.image_size
        image = self._generate_background(w, h)
        labels = []

        n_objects = random.randint(self.min_objects, self.max_objects)

        for _ in range(n_objects):
            class_id = random.randint(0, len(self.OBJECT_CLASSES) - 1)
            obj_size = random.randint(self.min_obj_size, self.max_obj_size)

            # random position (ensure object fits)
            x1 = random.randint(0, max(0, w - obj_size))
            y1 = random.randint(0, max(0, h - obj_size))
            x2 = min(x1 + obj_size, w)
            y2 = min(y1 + obj_size + random.randint(-20, 20), h)

            color = self._random_color()
            image = self._draw_object(image, class_id, x1, y1, x2, y2, color)

            labels.append({
                "class_id": class_id,
                "class_name": self.OBJECT_CLASSES[class_id],
                "bbox": [x1, y1, x2, y2],
            })

        # convert to YOLO format
        yolo_labels = []
        for lbl in labels:
            x1, y1, x2, y2 = lbl["bbox"]
            cx = ((x1 + x2) / 2) / w
            cy = ((y1 + y2) / 2) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            yolo_labels.append(f"{lbl['class_id']} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        return {
            "image": image,
            "labels": labels,
            "yolo_labels": yolo_labels,
        }

    def generate_batch(self, n: int, start_seed: int = 0) -> list[dict]:
        """Generate a batch of synthetic scenes."""
        return [self.generate(seed=start_seed + i) for i in range(n)]

    def save_dataset(self, output_dir: str, n_images: int, start_seed: int = 0):
        """Generate and save a dataset in YOLO format."""
        out = Path(output_dir)
        img_dir = out / "images"
        lbl_dir = out / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for i in range(n_images):
            scene = self.generate(seed=start_seed + i)
            img_path = img_dir / f"{i:06d}.jpg"
            lbl_path = lbl_dir / f"{i:06d}.txt"

            cv2.imwrite(str(img_path), scene["image"])
            with open(lbl_path, "w") as f:
                f.write("\n".join(scene["yolo_labels"]))

        # write class names
        with open(out / "classes.txt", "w") as f:
            for cls_id in sorted(self.OBJECT_CLASSES):
                f.write(self.OBJECT_CLASSES[cls_id] + "\n")

        print(f"Saved {n_images} images to {output_dir}")

    def _generate_background(self, w: int, h: int) -> np.ndarray:
        """Generate a randomized background."""
        bg_type = random.choice(["solid", "gradient", "noise", "textured"])

        if bg_type == "solid":
            color = self._random_color()
            bg = np.full((h, w, 3), color, dtype=np.uint8)
        elif bg_type == "gradient":
            c1 = self._random_color()
            c2 = self._random_color()
            bg = np.zeros((h, w, 3), dtype=np.uint8)
            for i in range(h):
                ratio = i / h
                bg[i] = np.array(c1) * (1 - ratio) + np.array(c2) * ratio
        elif bg_type == "noise":
            bg = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
            bg = cv2.GaussianBlur(bg, (15, 15), 0)
        else:
            # textured: circles/lines pattern
            base_color = self._random_color()
            bg = np.full((h, w, 3), base_color, dtype=np.uint8)
            for _ in range(random.randint(5, 20)):
                pt = (random.randint(0, w), random.randint(0, h))
                r = random.randint(10, 80)
                c = self._random_color()
                cv2.circle(bg, pt, r, c, -1)
            bg = cv2.GaussianBlur(bg, (21, 21), 0)

        return bg

    def _draw_object(self, image: np.ndarray, class_id: int,
                     x1: int, y1: int, x2: int, y2: int,
                     color: tuple) -> np.ndarray:
        """Draw a geometric object on the image."""
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        w = x2 - x1
        h = y2 - y1

        if class_id == 0:  # circle
            r = min(w, h) // 2
            cv2.circle(image, (cx, cy), r, color, -1)
            edge = tuple(max(0, c - 40) for c in color)
            cv2.circle(image, (cx, cy), r, edge, 2)
        elif class_id == 1:  # rectangle
            cv2.rectangle(image, (x1, y1), (x2, y2), color, -1)
            edge = tuple(max(0, c - 40) for c in color)
            cv2.rectangle(image, (x1, y1), (x2, y2), edge, 2)
        elif class_id == 2:  # triangle
            pts = np.array([
                [cx, y1], [x1, y2], [x2, y2]
            ], np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(image, [pts], color)
            cv2.polylines(image, [pts], True, tuple(max(0, c - 40) for c in color), 2)
        elif class_id == 3:  # ellipse
            axes = (w // 2, h // 2)
            cv2.ellipse(image, (cx, cy), axes, 0, 0, 360, color, -1)
            cv2.ellipse(image, (cx, cy), axes, 0, 0, 360,
                        tuple(max(0, c - 40) for c in color), 2)
        elif class_id == 4:  # pentagon
            angles = np.linspace(0, 2 * np.pi, 6)[:-1] - np.pi / 2
            r = min(w, h) // 2
            pts = np.array([
                [int(cx + r * np.cos(a)), int(cy + r * np.sin(a))]
                for a in angles
            ], np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(image, [pts], color)
            cv2.polylines(image, [pts], True, tuple(max(0, c - 40) for c in color), 2)

        return image

    @staticmethod
    def _random_color() -> tuple[int, int, int]:
        return (random.randint(30, 255), random.randint(30, 255), random.randint(30, 255))
