"""MOT evaluation metrics: MOTA, IDF1, and HOTA."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def compute_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Compute IoU between two boxes [x1,y1,x2,y2]."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])

    return inter / (area_a + area_b - inter + 1e-6)


class MOTMetrics:
    """Accumulates per-frame MOT metrics and computes MOTA/IDF1/HOTA."""

    def __init__(self, iou_threshold: float = 0.5):
        self.iou_threshold = iou_threshold
        self.frames: list[dict] = []

    def add_frame(self, gt_boxes: np.ndarray, gt_ids: np.ndarray,
                  pred_boxes: np.ndarray, pred_ids: np.ndarray):
        """Add a single frame of ground truth and predictions."""
        self.frames.append({
            "gt_boxes": gt_boxes,
            "gt_ids": gt_ids,
            "pred_boxes": pred_boxes,
            "pred_ids": pred_ids,
        })

    def compute_mota(self) -> dict:
        """
        Compute MOTA (Multi-Object Tracking Accuracy).

        MOTA = 1 - (FN + FP + IDSW) / total_gt
        """
        total_gt = 0
        total_fp = 0
        total_fn = 0
        total_idsw = 0

        prev_matches = {}  # gt_id -> pred_id

        for frame in self.frames:
            gt_boxes = frame["gt_boxes"]
            gt_ids = frame["gt_ids"]
            pred_boxes = frame["pred_boxes"]
            pred_ids = frame["pred_ids"]

            total_gt += len(gt_boxes)

            if len(gt_boxes) == 0:
                total_fp += len(pred_boxes)
                continue
            if len(pred_boxes) == 0:
                total_fn += len(gt_boxes)
                continue

            # compute cost matrix
            cost = np.zeros((len(gt_boxes), len(pred_boxes)))
            for i in range(len(gt_boxes)):
                for j in range(len(pred_boxes)):
                    cost[i, j] = 1.0 - compute_iou(gt_boxes[i], pred_boxes[j])

            row_idx, col_idx = linear_sum_assignment(cost)

            matched_gt = set()
            matched_pred = set()
            curr_matches = {}

            for r, c in zip(row_idx, col_idx):
                iou = 1.0 - cost[r, c]
                if iou >= self.iou_threshold:
                    matched_gt.add(r)
                    matched_pred.add(c)
                    gt_id = gt_ids[r] if r < len(gt_ids) else -1
                    pred_id = pred_ids[c] if c < len(pred_ids) else -1
                    curr_matches[gt_id] = pred_id

                    if gt_id in prev_matches and prev_matches[gt_id] != pred_id:
                        total_idsw += 1

            total_fn += len(gt_boxes) - len(matched_gt)
            total_fp += len(pred_boxes) - len(matched_pred)
            prev_matches = curr_matches

        mota = 1.0 - (total_fn + total_fp + total_idsw) / max(total_gt, 1)

        return {
            "mota": float(mota),
            "fn": total_fn,
            "fp": total_fp,
            "idsw": total_idsw,
            "total_gt": total_gt,
        }

    def compute_idf1(self) -> dict:
        """
        Compute IDF1 (ID F1-score).

        Measures identity preservation across frames.
        """
        gt_id_counts = {}
        pred_id_counts = {}
        match_counts = {}

        for frame in self.frames:
            gt_boxes = frame["gt_boxes"]
            gt_ids = frame["gt_ids"]
            pred_boxes = frame["pred_boxes"]
            pred_ids = frame["pred_ids"]

            for gid in gt_ids:
                gt_id_counts[gid] = gt_id_counts.get(gid, 0) + 1

            for pid in pred_ids:
                pred_id_counts[pid] = pred_id_counts.get(pid, 0) + 1

            if len(gt_boxes) == 0 or len(pred_boxes) == 0:
                continue

            cost = np.zeros((len(gt_boxes), len(pred_boxes)))
            for i in range(len(gt_boxes)):
                for j in range(len(pred_boxes)):
                    cost[i, j] = 1.0 - compute_iou(gt_boxes[i], pred_boxes[j])

            row_idx, col_idx = linear_sum_assignment(cost)

            for r, c in zip(row_idx, col_idx):
                if 1.0 - cost[r, c] >= self.iou_threshold:
                    key = (gt_ids[r], pred_ids[c])
                    match_counts[key] = match_counts.get(key, 0) + 1

        # find best matching between gt and pred IDs
        all_gt_ids = list(gt_id_counts.keys())
        all_pred_ids = list(pred_id_counts.keys())

        if not all_gt_ids or not all_pred_ids:
            return {"idf1": 0.0, "idp": 0.0, "idr": 0.0}

        match_matrix = np.zeros((len(all_gt_ids), len(all_pred_ids)))
        for i, gid in enumerate(all_gt_ids):
            for j, pid in enumerate(all_pred_ids):
                match_matrix[i, j] = match_counts.get((gid, pid), 0)

        row_idx, col_idx = linear_sum_assignment(-match_matrix)

        idtp = sum(match_matrix[r, c] for r, c in zip(row_idx, col_idx))
        total_gt_dets = sum(gt_id_counts.values())
        total_pred_dets = sum(pred_id_counts.values())

        idr = idtp / max(total_gt_dets, 1)
        idp = idtp / max(total_pred_dets, 1)
        idf1 = 2 * idtp / max(total_gt_dets + total_pred_dets, 1)

        return {"idf1": float(idf1), "idp": float(idp), "idr": float(idr)}

    def compute_hota(self, alpha: float = 0.5) -> dict:
        """
        Compute HOTA (Higher Order Tracking Accuracy) at a single alpha.

        HOTA = sqrt(DetA * AssA)
        """
        tp = 0
        fn = 0
        fp = 0
        ass_sum = 0.0

        for frame in self.frames:
            gt_boxes = frame["gt_boxes"]
            pred_boxes = frame["pred_boxes"]

            if len(gt_boxes) == 0:
                fp += len(pred_boxes)
                continue
            if len(pred_boxes) == 0:
                fn += len(gt_boxes)
                continue

            cost = np.zeros((len(gt_boxes), len(pred_boxes)))
            for i in range(len(gt_boxes)):
                for j in range(len(pred_boxes)):
                    cost[i, j] = 1.0 - compute_iou(gt_boxes[i], pred_boxes[j])

            row_idx, col_idx = linear_sum_assignment(cost)

            matched = 0
            for r, c in zip(row_idx, col_idx):
                if 1.0 - cost[r, c] >= alpha:
                    matched += 1
                    ass_sum += 1.0 - cost[r, c]

            tp += matched
            fn += len(gt_boxes) - matched
            fp += len(pred_boxes) - matched

        det_a = tp / max(tp + fn + fp, 1)
        ass_a = ass_sum / max(tp, 1) if tp > 0 else 0
        hota = np.sqrt(det_a * ass_a)

        return {
            "hota": float(hota),
            "det_a": float(det_a),
            "ass_a": float(ass_a),
            "tp": tp, "fn": fn, "fp": fp,
        }

    def compute_all(self) -> dict:
        """Compute all metrics."""
        mota_result = self.compute_mota()
        idf1_result = self.compute_idf1()
        hota_result = self.compute_hota()
        return {**mota_result, **idf1_result, **hota_result}
