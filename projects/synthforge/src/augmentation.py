"""Modular augmentation pipeline with robotics-specific transforms."""

from __future__ import annotations

import random
from typing import Any

import albumentations as A
import cv2
import numpy as np


def get_standard_pipeline(image_size: tuple[int, int] = (640, 640)) -> A.Compose:
    """Standard augmentation pipeline for object detection training."""
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=0.3),
        A.GaussianBlur(blur_limit=(3, 7), p=0.2),
        A.GaussNoise(p=0.2),
        A.CLAHE(clip_limit=2.0, p=0.2),
        A.Resize(image_size[0], image_size[1]),
    ], bbox_params=A.BboxParams(
        format="pascal_voc", label_fields=["class_ids"],
        min_visibility=0.3,
    ))


def get_heavy_pipeline(image_size: tuple[int, int] = (640, 640)) -> A.Compose:
    """Aggressive augmentation for data-scarce scenarios."""
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.RandomRotate90(p=0.2),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=15, p=0.5),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3),
            A.RandomGamma(gamma_limit=(70, 130)),
        ], p=0.5),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7)),
            A.MotionBlur(blur_limit=(3, 7)),
            A.MedianBlur(blur_limit=5),
        ], p=0.3),
        A.OneOf([
            A.GaussNoise(),
            A.ISONoise(),
        ], p=0.2),
        A.RandomShadow(num_shadows_lower=1, num_shadows_upper=3, p=0.2),
        A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=0.1),
        A.ImageCompression(quality_lower=50, quality_upper=95, p=0.2),
        A.Resize(image_size[0], image_size[1]),
    ], bbox_params=A.BboxParams(
        format="pascal_voc", label_fields=["class_ids"],
        min_visibility=0.3,
    ))


def apply_copy_paste(
    base_image: np.ndarray,
    base_labels: list[dict],
    donor_image: np.ndarray,
    donor_labels: list[dict],
    max_paste: int = 3,
) -> tuple[np.ndarray, list[dict]]:
    """
    Copy-paste augmentation for rare object injection.

    Crops objects from donor image and pastes them into base image
    at random positions to increase rare class representation.
    """
    result = base_image.copy()
    result_labels = list(base_labels)
    h, w = result.shape[:2]

    if not donor_labels:
        return result, result_labels

    n_paste = min(random.randint(1, max_paste), len(donor_labels))
    selected = random.sample(donor_labels, n_paste)

    for lbl in selected:
        x1, y1, x2, y2 = lbl["bbox"]
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(donor_image.shape[1], x2), min(donor_image.shape[0], y2)

        crop = donor_image[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        ch, cw = crop.shape[:2]

        # random position in base image
        paste_x = random.randint(0, max(0, w - cw))
        paste_y = random.randint(0, max(0, h - ch))

        # simple alpha blending at edges
        mask = np.ones((ch, cw), dtype=np.float32)
        border = min(5, ch // 4, cw // 4)
        if border > 0:
            mask[:border, :] *= np.linspace(0, 1, border)[:, None]
            mask[-border:, :] *= np.linspace(1, 0, border)[:, None]
            mask[:, :border] *= np.linspace(0, 1, border)[None, :]
            mask[:, -border:] *= np.linspace(1, 0, border)[None, :]

        roi = result[paste_y:paste_y + ch, paste_x:paste_x + cw]
        if roi.shape[:2] != (ch, cw):
            continue

        for c in range(3):
            roi[:, :, c] = (
                crop[:, :, c] * mask + roi[:, :, c] * (1 - mask)
            ).astype(np.uint8)
        result[paste_y:paste_y + ch, paste_x:paste_x + cw] = roi

        result_labels.append({
            "class_id": lbl["class_id"],
            "class_name": lbl.get("class_name", ""),
            "bbox": [paste_x, paste_y, paste_x + cw, paste_y + ch],
        })

    return result, result_labels


def apply_depth_noise(image: np.ndarray, depth_map: np.ndarray | None = None
                      ) -> np.ndarray:
    """
    Apply depth-dependent noise to simulate RGB-D sensor artifacts.

    Objects further from camera get more noise (simulating real sensor behavior).
    If no depth map provided, generates a synthetic one.
    """
    h, w = image.shape[:2]

    if depth_map is None:
        # synthetic depth: top=far, bottom=near
        depth_map = np.linspace(0.3, 1.0, h)[:, None] * np.ones((1, w))
        depth_map = depth_map.astype(np.float32)

    # noise proportional to depth
    noise_scale = depth_map * 15
    noise = np.random.normal(0, 1, image.shape) * noise_scale[:, :, None]
    result = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return result
