"""Auto-labeling pipeline using pre-trained detection models."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


class AutoLabeler:
    """
    Auto-label real images using a pre-trained YOLO model.

    Routes high-confidence predictions directly to dataset and
    flags low-confidence ones for human review.
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        high_confidence: float = 0.7,
        low_confidence: float = 0.3,
    ):
        self.model = YOLO(model_path)
        self.high_confidence = high_confidence
        self.low_confidence = low_confidence

    def label_image(self, image_path: str) -> dict:
        """
        Auto-label a single image.

        Returns:
            dict with:
                auto_labels: high-confidence labels (ready for dataset)
                review_labels: low-confidence labels (need human review)
                rejected: very low confidence (discarded)
        """
        results = self.model(image_path, verbose=False)
        boxes = results[0].boxes

        if len(boxes) == 0:
            return {"auto_labels": [], "review_labels": [], "rejected": []}

        image = cv2.imread(image_path)
        h, w = image.shape[:2]

        bboxes = boxes.xyxy.cpu().numpy()
        scores = boxes.conf.cpu().numpy()
        class_ids = boxes.cls.cpu().numpy().astype(int)
        class_names = [results[0].names[c] for c in class_ids]

        auto_labels = []
        review_labels = []
        rejected = []

        for i in range(len(bboxes)):
            label = {
                "bbox": bboxes[i].tolist(),
                "score": float(scores[i]),
                "class_id": int(class_ids[i]),
                "class_name": class_names[i],
                "yolo_format": self._to_yolo(bboxes[i], w, h, class_ids[i]),
            }

            if scores[i] >= self.high_confidence:
                label["status"] = "auto_approved"
                auto_labels.append(label)
            elif scores[i] >= self.low_confidence:
                label["status"] = "needs_review"
                review_labels.append(label)
            else:
                label["status"] = "rejected"
                rejected.append(label)

        return {
            "auto_labels": auto_labels,
            "review_labels": review_labels,
            "rejected": rejected,
        }

    def label_directory(self, input_dir: str, output_dir: str) -> dict:
        """
        Auto-label all images in a directory.

        Saves approved labels to output_dir/labels/
        and review candidates to output_dir/review/
        """
        in_path = Path(input_dir)
        out_path = Path(output_dir)
        labels_dir = out_path / "labels"
        review_dir = out_path / "review"
        labels_dir.mkdir(parents=True, exist_ok=True)
        review_dir.mkdir(parents=True, exist_ok=True)

        stats = {
            "total_images": 0,
            "total_auto": 0,
            "total_review": 0,
            "total_rejected": 0,
        }

        image_exts = {".jpg", ".jpeg", ".png", ".bmp"}
        images = [f for f in in_path.iterdir()
                  if f.suffix.lower() in image_exts]

        for img_path in sorted(images):
            result = self.label_image(str(img_path))
            stats["total_images"] += 1
            stats["total_auto"] += len(result["auto_labels"])
            stats["total_review"] += len(result["review_labels"])
            stats["total_rejected"] += len(result["rejected"])

            # save auto-approved labels
            if result["auto_labels"]:
                lbl_path = labels_dir / f"{img_path.stem}.txt"
                with open(lbl_path, "w") as f:
                    for lbl in result["auto_labels"]:
                        f.write(lbl["yolo_format"] + "\n")

            # save review candidates with confidence info
            if result["review_labels"]:
                rev_path = review_dir / f"{img_path.stem}.txt"
                with open(rev_path, "w") as f:
                    for lbl in result["review_labels"]:
                        f.write(f"# score={lbl['score']:.3f} class={lbl['class_name']}\n")
                        f.write(lbl["yolo_format"] + "\n")

        return stats

    @staticmethod
    def _to_yolo(bbox: np.ndarray, img_w: int, img_h: int, class_id: int) -> str:
        x1, y1, x2, y2 = bbox
        cx = ((x1 + x2) / 2) / img_w
        cy = ((y1 + y2) / 2) / img_h
        w = (x2 - x1) / img_w
        h = (y2 - y1) / img_h
        return f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
