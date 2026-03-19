"""Domain randomization for synthetic data generation."""

from __future__ import annotations

import random

import cv2
import numpy as np


class DomainRandomizer:
    """
    Applies domain randomization to synthetic scenes.

    Randomizes lighting, material appearance, camera effects,
    and environmental conditions to improve sim-to-real transfer.
    """

    def __init__(self, config: dict | None = None):
        self.config = config or self._default_config()

    @staticmethod
    def _default_config() -> dict:
        return {
            "brightness": {"enabled": True, "range": (-50, 50)},
            "contrast": {"enabled": True, "range": (0.7, 1.3)},
            "hue_shift": {"enabled": True, "range": (-15, 15)},
            "blur": {"enabled": True, "kernel_range": (1, 5)},
            "noise": {"enabled": True, "sigma_range": (0, 25)},
            "shadow": {"enabled": True, "max_shadows": 3},
            "rotation": {"enabled": True, "range": (-5, 5)},
            "scale": {"enabled": True, "range": (0.9, 1.1)},
            "jpeg_quality": {"enabled": True, "range": (40, 95)},
            "motion_blur": {"enabled": True, "kernel_range": (3, 9)},
        }

    def apply(self, image: np.ndarray, labels: list[dict] | None = None
              ) -> tuple[np.ndarray, list[dict] | None]:
        """
        Apply random domain randomization transforms.

        Args:
            image: input BGR image
            labels: optional list of label dicts with bbox [x1,y1,x2,y2]

        Returns:
            (randomized_image, updated_labels)
        """
        h, w = image.shape[:2]
        result = image.copy()

        if self._enabled("brightness"):
            lo, hi = self.config["brightness"]["range"]
            delta = random.randint(lo, hi)
            result = np.clip(result.astype(np.int16) + delta, 0, 255).astype(np.uint8)

        if self._enabled("contrast"):
            lo, hi = self.config["contrast"]["range"]
            alpha = random.uniform(lo, hi)
            result = np.clip(result.astype(np.float32) * alpha, 0, 255).astype(np.uint8)

        if self._enabled("hue_shift"):
            lo, hi = self.config["hue_shift"]["range"]
            shift = random.randint(lo, hi)
            hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
            hsv[:, :, 0] = (hsv[:, :, 0].astype(int) + shift) % 180
            result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        if self._enabled("noise"):
            lo, hi = self.config["noise"]["sigma_range"]
            sigma = random.uniform(lo, hi)
            if sigma > 0:
                noise = np.random.normal(0, sigma, result.shape).astype(np.int16)
                result = np.clip(result.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        if self._enabled("shadow"):
            result = self._add_shadows(result)

        if self._enabled("blur") and random.random() < 0.3:
            lo, hi = self.config["blur"]["kernel_range"]
            k = random.choice(range(lo, hi + 1, 2))
            if k > 1:
                result = cv2.GaussianBlur(result, (k, k), 0)

        if self._enabled("motion_blur") and random.random() < 0.2:
            result = self._apply_motion_blur(result)

        if self._enabled("jpeg_quality") and random.random() < 0.3:
            lo, hi = self.config["jpeg_quality"]["range"]
            quality = random.randint(lo, hi)
            _, encoded = cv2.imencode(".jpg", result, [cv2.IMWRITE_JPEG_QUALITY, quality])
            result = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

        if self._enabled("rotation"):
            result, labels = self._apply_rotation(result, labels)

        return result, labels

    def _enabled(self, key: str) -> bool:
        return self.config.get(key, {}).get("enabled", False) and random.random() < 0.7

    def _add_shadows(self, image: np.ndarray) -> np.ndarray:
        """Add random shadow regions to simulate lighting variation."""
        h, w = image.shape[:2]
        result = image.copy()
        n_shadows = random.randint(1, self.config["shadow"]["max_shadows"])

        for _ in range(n_shadows):
            # random polygon shadow
            n_pts = random.randint(3, 6)
            pts = np.array([
                [random.randint(0, w), random.randint(0, h)]
                for _ in range(n_pts)
            ], np.int32)

            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [pts], 255)
            mask = cv2.GaussianBlur(mask, (31, 31), 0)

            darkness = random.uniform(0.4, 0.8)
            shadow = result.astype(np.float32)
            shadow_factor = 1.0 - (1.0 - darkness) * (mask.astype(np.float32) / 255.0)
            for c in range(3):
                shadow[:, :, c] *= shadow_factor
            result = np.clip(shadow, 0, 255).astype(np.uint8)

        return result

    def _apply_motion_blur(self, image: np.ndarray) -> np.ndarray:
        """Simulate motion blur from camera/robot movement."""
        lo, hi = self.config["motion_blur"]["kernel_range"]
        k = random.choice(range(lo, hi + 1, 2))
        if k < 3:
            return image

        direction = random.choice(["horizontal", "vertical", "diagonal"])
        kernel = np.zeros((k, k))

        if direction == "horizontal":
            kernel[k // 2, :] = 1.0
        elif direction == "vertical":
            kernel[:, k // 2] = 1.0
        else:
            np.fill_diagonal(kernel, 1.0)

        kernel /= kernel.sum()
        return cv2.filter2D(image, -1, kernel)

    def _apply_rotation(self, image: np.ndarray, labels: list[dict] | None
                        ) -> tuple[np.ndarray, list[dict] | None]:
        """Apply small rotation with bbox adjustment."""
        lo, hi = self.config["rotation"]["range"]
        angle = random.uniform(lo, hi)

        h, w = image.shape[:2]
        cx, cy = w / 2, h / 2
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)

        if labels:
            for lbl in labels:
                x1, y1, x2, y2 = lbl["bbox"]
                corners = np.array([
                    [x1, y1, 1], [x2, y1, 1],
                    [x1, y2, 1], [x2, y2, 1]
                ], dtype=np.float64)
                transformed = (M @ corners.T).T
                lbl["bbox"] = [
                    max(0, int(transformed[:, 0].min())),
                    max(0, int(transformed[:, 1].min())),
                    min(w, int(transformed[:, 0].max())),
                    min(h, int(transformed[:, 1].max())),
                ]

        return rotated, labels
