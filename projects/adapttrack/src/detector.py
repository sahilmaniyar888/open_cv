"""Object detector with MC Dropout uncertainty estimation."""

from __future__ import annotations

import numpy as np
import torch
from ultralytics import YOLO


class UncertaintyDetector:
    """
    YOLOv8 detector with Monte Carlo Dropout for uncertainty estimation.

    Runs multiple forward passes with dropout enabled to estimate
    per-detection uncertainty from prediction variance.
    """

    def __init__(self, model_path: str = "yolov8n.pt", confidence: float = 0.25,
                 mc_samples: int = 5, device: str | None = None):
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.mc_samples = mc_samples

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

    def _enable_dropout(self):
        """Enable dropout layers during inference for MC sampling."""
        for module in self.model.model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.train()

    def _disable_dropout(self):
        """Restore dropout layers to eval mode."""
        for module in self.model.model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.eval()

    def detect(self, frame: np.ndarray) -> dict:
        """
        Run detection on a single frame.

        Returns dict with keys:
            bboxes: np.ndarray (N, 4) [x1, y1, x2, y2]
            scores: np.ndarray (N,)
            class_ids: np.ndarray (N,) int
            uncertainties: np.ndarray (N,) [0-1], higher = more uncertain
        """
        results = self.model(frame, conf=self.confidence, verbose=False)
        boxes = results[0].boxes

        if len(boxes) == 0:
            return {
                "bboxes": np.empty((0, 4)),
                "scores": np.empty(0),
                "class_ids": np.empty(0, dtype=int),
                "uncertainties": np.empty(0),
            }

        bboxes = boxes.xyxy.cpu().numpy()
        scores = boxes.conf.cpu().numpy()
        class_ids = boxes.cls.cpu().numpy().astype(int)

        return {
            "bboxes": bboxes,
            "scores": scores,
            "class_ids": class_ids,
            "uncertainties": self._estimate_uncertainty(frame, bboxes, scores),
        }

    def _estimate_uncertainty(self, frame: np.ndarray, base_bboxes: np.ndarray,
                              base_scores: np.ndarray) -> np.ndarray:
        """
        Estimate detection uncertainty via MC Dropout.

        Runs multiple forward passes and measures variance in predictions.
        For models without dropout layers, uses score-based approximation.
        """
        if self.mc_samples <= 1:
            return 1.0 - base_scores

        has_dropout = any(
            isinstance(m, torch.nn.Dropout)
            for m in self.model.model.modules()
        )

        if not has_dropout:
            # fallback: use score variance from augmented inference
            return self._augmented_uncertainty(frame, base_bboxes, base_scores)

        self._enable_dropout()
        all_scores = []

        try:
            for _ in range(self.mc_samples):
                results = self.model(frame, conf=self.confidence * 0.5, verbose=False)
                mc_boxes = results[0].boxes
                if len(mc_boxes) > 0:
                    mc_bboxes = mc_boxes.xyxy.cpu().numpy()
                    mc_scores = mc_boxes.conf.cpu().numpy()
                    matched_scores = self._match_mc_scores(
                        base_bboxes, mc_bboxes, mc_scores
                    )
                    all_scores.append(matched_scores)
                else:
                    all_scores.append(np.zeros(len(base_bboxes)))
        finally:
            self._disable_dropout()

        if len(all_scores) < 2:
            return 1.0 - base_scores

        score_matrix = np.array(all_scores)
        variance = np.var(score_matrix, axis=0)
        # normalize variance to [0, 1]
        max_var = np.max(variance) if np.max(variance) > 0 else 1.0
        uncertainty = variance / max_var
        return np.clip(uncertainty, 0, 1)

    def _augmented_uncertainty(self, frame: np.ndarray, base_bboxes: np.ndarray,
                               base_scores: np.ndarray) -> np.ndarray:
        """Estimate uncertainty through minor input perturbations."""
        all_scores = [base_scores.copy()]

        for i in range(min(self.mc_samples, 3)):
            # small brightness/contrast perturbation
            alpha = 1.0 + np.random.uniform(-0.1, 0.1)
            beta = np.random.uniform(-10, 10)
            perturbed = np.clip(alpha * frame + beta, 0, 255).astype(np.uint8)

            results = self.model(perturbed, conf=self.confidence * 0.5, verbose=False)
            p_boxes = results[0].boxes
            if len(p_boxes) > 0:
                p_bboxes = p_boxes.xyxy.cpu().numpy()
                p_scores = p_boxes.conf.cpu().numpy()
                matched = self._match_mc_scores(base_bboxes, p_bboxes, p_scores)
                all_scores.append(matched)
            else:
                all_scores.append(np.zeros(len(base_bboxes)))

        score_matrix = np.array(all_scores)
        variance = np.var(score_matrix, axis=0)
        # combine variance with inverse score for robust uncertainty
        uncertainty = 0.5 * (1.0 - base_scores) + 0.5 * np.clip(
            variance / (np.max(variance) + 1e-6), 0, 1
        )
        return np.clip(uncertainty, 0, 1)

    def _match_mc_scores(self, base_bboxes: np.ndarray,
                         mc_bboxes: np.ndarray,
                         mc_scores: np.ndarray) -> np.ndarray:
        """Match MC sample detections back to base detections by IoU."""
        from .association import iou_batch

        if len(mc_bboxes) == 0:
            return np.zeros(len(base_bboxes))

        iou = iou_batch(base_bboxes, mc_bboxes)
        matched_scores = np.zeros(len(base_bboxes))

        for i in range(len(base_bboxes)):
            best_match = np.argmax(iou[i])
            if iou[i, best_match] > 0.3:
                matched_scores[i] = mc_scores[best_match]

        return matched_scores
