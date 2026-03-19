"""Config-driven evaluation harness for reproducible tracking experiments."""

from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np

from .metrics import MOTMetrics


class EvaluationHarness:
    """
    Runs tracker evaluation with configurable parameters.

    Supports MOT-format ground truth annotations and produces
    reproducible benchmark results.
    """

    def __init__(self, config: dict | None = None):
        self.config = config or self._default_config()

    @staticmethod
    def _default_config() -> dict:
        return {
            "iou_threshold": 0.5,
            "confidence_threshold": 0.25,
            "max_age": 30,
            "min_hits": 3,
            "high_thresh": 0.5,
            "low_thresh": 0.1,
        }

    @classmethod
    def from_config_file(cls, path: str) -> EvaluationHarness:
        with open(path) as f:
            config = json.load(f)
        return cls(config)

    def evaluate_sequence(self, tracker, detector, video_path: str,
                          gt_path: str | None = None) -> dict:
        """
        Run tracker on a video sequence and evaluate.

        Args:
            tracker: AdaptTracker instance
            detector: UncertaintyDetector instance
            video_path: path to video file
            gt_path: optional path to MOT-format ground truth

        Returns:
            dict with metrics and timing info
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        gt_data = self._load_ground_truth(gt_path) if gt_path else None
        metrics = MOTMetrics(iou_threshold=self.config["iou_threshold"])
        timings = []

        frame_idx = 0
        all_tracks = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            t0 = time.time()
            detections = detector.detect(frame)
            tracks = tracker.update(detections)
            elapsed = time.time() - t0
            timings.append(elapsed)

            if gt_data and frame_idx in gt_data:
                gt_frame = gt_data[frame_idx]
                pred_boxes = np.array([t["bbox"] for t in tracks]) if tracks else np.empty((0, 4))
                pred_ids = np.array([t["track_id"] for t in tracks]) if tracks else np.empty(0, dtype=int)
                metrics.add_frame(
                    gt_frame["boxes"], gt_frame["ids"],
                    pred_boxes, pred_ids
                )

            for t in tracks:
                all_tracks.append({"frame": frame_idx, **t})

            frame_idx += 1

        cap.release()

        result = {
            "num_frames": frame_idx,
            "avg_fps": 1.0 / np.mean(timings) if timings else 0,
            "avg_latency_ms": np.mean(timings) * 1000 if timings else 0,
            "tracker_stats": tracker.get_stats(),
            "config": self.config,
        }

        if gt_data:
            result["metrics"] = metrics.compute_all()

        return result

    def evaluate_synthetic(self, tracker, detections_list: list[dict],
                           gt_list: list[dict] | None = None) -> dict:
        """Evaluate tracker on pre-computed detections (for testing)."""
        metrics = MOTMetrics(iou_threshold=self.config["iou_threshold"])
        timings = []

        for frame_idx, detections in enumerate(detections_list):
            t0 = time.time()
            tracks = tracker.update(detections)
            elapsed = time.time() - t0
            timings.append(elapsed)

            if gt_list and frame_idx < len(gt_list):
                gt = gt_list[frame_idx]
                pred_boxes = np.array([t["bbox"] for t in tracks]) if tracks else np.empty((0, 4))
                pred_ids = np.array([t["track_id"] for t in tracks]) if tracks else np.empty(0, dtype=int)
                metrics.add_frame(
                    gt["boxes"], gt["ids"],
                    pred_boxes, pred_ids
                )

        result = {
            "num_frames": len(detections_list),
            "avg_fps": 1.0 / np.mean(timings) if timings else 0,
            "tracker_stats": tracker.get_stats(),
        }

        if gt_list:
            result["metrics"] = metrics.compute_all()

        return result

    def _load_ground_truth(self, gt_path: str) -> dict:
        """
        Load MOT-format ground truth.

        Format: frame_id, track_id, x, y, w, h, conf, class, visibility
        """
        gt_data = {}
        path = Path(gt_path)

        if not path.exists():
            return gt_data

        with open(path) as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 6:
                    continue
                frame_id = int(parts[0]) - 1  # MOT is 1-indexed
                track_id = int(parts[1])
                x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
                bbox = np.array([x, y, x + w, y + h])

                if frame_id not in gt_data:
                    gt_data[frame_id] = {"boxes": [], "ids": []}
                gt_data[frame_id]["boxes"].append(bbox)
                gt_data[frame_id]["ids"].append(track_id)

        # convert lists to arrays
        for frame_id in gt_data:
            gt_data[frame_id]["boxes"] = np.array(gt_data[frame_id]["boxes"])
            gt_data[frame_id]["ids"] = np.array(gt_data[frame_id]["ids"])

        return gt_data
