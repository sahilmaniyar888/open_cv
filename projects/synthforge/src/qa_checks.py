"""Automated quality assurance for dataset labels and images."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import cv2
import numpy as np


class DatasetQA:
    """
    Runs automated quality checks on labeled datasets.

    Validates label format, bounding box sanity, class balance,
    and image integrity.
    """

    def __init__(self, image_dir: str, label_dir: str, num_classes: int = 5):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.num_classes = num_classes
        self.issues: list[dict] = []

    def run_all_checks(self) -> dict:
        """Run all QA checks and return summary."""
        self.issues = []

        self._check_image_label_alignment()
        self._check_label_format()
        self._check_bbox_sanity()
        self._check_class_balance()
        self._check_image_integrity()

        return self.get_summary()

    def _check_image_label_alignment(self):
        """Verify every image has a label file and vice versa."""
        image_exts = {".jpg", ".jpeg", ".png", ".bmp"}
        images = {f.stem for f in self.image_dir.iterdir()
                  if f.suffix.lower() in image_exts}
        labels = {f.stem for f in self.label_dir.iterdir()
                  if f.suffix == ".txt"}

        missing_labels = images - labels
        orphan_labels = labels - images

        for stem in missing_labels:
            self.issues.append({
                "type": "missing_label",
                "severity": "error",
                "file": stem,
                "message": f"image {stem} has no label file",
            })

        for stem in orphan_labels:
            self.issues.append({
                "type": "orphan_label",
                "severity": "warning",
                "file": stem,
                "message": f"label {stem}.txt has no matching image",
            })

    def _check_label_format(self):
        """Validate YOLO label format: class cx cy w h."""
        for lbl_file in sorted(self.label_dir.glob("*.txt")):
            with open(lbl_file) as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue

                    parts = line.split()
                    if len(parts) != 5:
                        self.issues.append({
                            "type": "format_error",
                            "severity": "error",
                            "file": lbl_file.name,
                            "line": line_num,
                            "message": f"expected 5 values, got {len(parts)}",
                        })
                        continue

                    try:
                        cls_id = int(parts[0])
                        values = [float(v) for v in parts[1:]]
                    except ValueError:
                        self.issues.append({
                            "type": "format_error",
                            "severity": "error",
                            "file": lbl_file.name,
                            "line": line_num,
                            "message": "non-numeric values in label",
                        })
                        continue

                    if cls_id < 0 or cls_id >= self.num_classes:
                        self.issues.append({
                            "type": "invalid_class",
                            "severity": "error",
                            "file": lbl_file.name,
                            "line": line_num,
                            "message": f"class {cls_id} outside range [0, {self.num_classes})",
                        })

    def _check_bbox_sanity(self):
        """Check for degenerate or out-of-bounds bounding boxes."""
        for lbl_file in sorted(self.label_dir.glob("*.txt")):
            with open(lbl_file) as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue

                    parts = line.split()
                    if len(parts) != 5:
                        continue

                    try:
                        cx, cy, w, h = [float(v) for v in parts[1:]]
                    except ValueError:
                        continue

                    # check normalized bounds
                    if not (0 <= cx <= 1 and 0 <= cy <= 1):
                        self.issues.append({
                            "type": "bbox_oob",
                            "severity": "warning",
                            "file": lbl_file.name,
                            "line": line_num,
                            "message": f"center ({cx:.3f}, {cy:.3f}) outside [0,1]",
                        })

                    if w <= 0 or h <= 0:
                        self.issues.append({
                            "type": "bbox_degenerate",
                            "severity": "error",
                            "file": lbl_file.name,
                            "line": line_num,
                            "message": f"zero/negative size ({w:.3f}, {h:.3f})",
                        })

                    if w > 1 or h > 1:
                        self.issues.append({
                            "type": "bbox_oversized",
                            "severity": "warning",
                            "file": lbl_file.name,
                            "line": line_num,
                            "message": f"size ({w:.3f}, {h:.3f}) exceeds image bounds",
                        })

                    # tiny objects
                    if w * h < 0.001:
                        self.issues.append({
                            "type": "bbox_tiny",
                            "severity": "info",
                            "file": lbl_file.name,
                            "line": line_num,
                            "message": f"very small object (area={w * h:.5f})",
                        })

    def _check_class_balance(self):
        """Check class distribution for severe imbalance."""
        class_counts = Counter()

        for lbl_file in self.label_dir.glob("*.txt"):
            with open(lbl_file) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split()
                    if len(parts) >= 1:
                        try:
                            class_counts[int(parts[0])] += 1
                        except ValueError:
                            pass

        if not class_counts:
            return

        total = sum(class_counts.values())
        expected = total / max(self.num_classes, 1)

        for cls_id in range(self.num_classes):
            count = class_counts.get(cls_id, 0)
            if count == 0:
                self.issues.append({
                    "type": "class_missing",
                    "severity": "warning",
                    "message": f"class {cls_id} has zero instances",
                })
            elif count < expected * 0.1:
                self.issues.append({
                    "type": "class_imbalance",
                    "severity": "warning",
                    "message": f"class {cls_id} severely underrepresented ({count}/{total})",
                })

    def _check_image_integrity(self, sample_size: int = 50):
        """Verify images can be loaded and have consistent dimensions."""
        image_exts = {".jpg", ".jpeg", ".png", ".bmp"}
        images = [f for f in self.image_dir.iterdir()
                  if f.suffix.lower() in image_exts]

        # sample for efficiency
        check_images = images[:sample_size] if len(images) > sample_size else images
        dims = []

        for img_path in check_images:
            img = cv2.imread(str(img_path))
            if img is None:
                self.issues.append({
                    "type": "corrupt_image",
                    "severity": "error",
                    "file": img_path.name,
                    "message": "image could not be loaded",
                })
            else:
                dims.append(img.shape[:2])

        if dims:
            unique_dims = set(dims)
            if len(unique_dims) > 1:
                self.issues.append({
                    "type": "inconsistent_dims",
                    "severity": "info",
                    "message": f"found {len(unique_dims)} different image sizes: {unique_dims}",
                })

    def get_summary(self) -> dict:
        """Get QA check summary."""
        severity_counts = Counter(i["severity"] for i in self.issues)
        type_counts = Counter(i["type"] for i in self.issues)

        return {
            "total_issues": len(self.issues),
            "errors": severity_counts.get("error", 0),
            "warnings": severity_counts.get("warning", 0),
            "info": severity_counts.get("info", 0),
            "issue_types": dict(type_counts),
            "passed": severity_counts.get("error", 0) == 0,
            "details": self.issues[:20],  # first 20 issues
        }
