"""Detection-to-track association using IoU and Hungarian algorithm."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def iou_batch(bboxes_a: np.ndarray, bboxes_b: np.ndarray) -> np.ndarray:
    """Compute pairwise IoU between two sets of bboxes [x1,y1,x2,y2]."""
    if len(bboxes_a) == 0 or len(bboxes_b) == 0:
        return np.empty((len(bboxes_a), len(bboxes_b)))

    x1 = np.maximum(bboxes_a[:, 0:1], bboxes_b[:, 0:1].T)
    y1 = np.maximum(bboxes_a[:, 1:2], bboxes_b[:, 1:2].T)
    x2 = np.minimum(bboxes_a[:, 2:3], bboxes_b[:, 2:3].T)
    y2 = np.minimum(bboxes_a[:, 3:4], bboxes_b[:, 3:4].T)

    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area_a = (bboxes_a[:, 2] - bboxes_a[:, 0]) * (bboxes_a[:, 3] - bboxes_a[:, 1])
    area_b = (bboxes_b[:, 2] - bboxes_b[:, 0]) * (bboxes_b[:, 3] - bboxes_b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter

    return inter / (union + 1e-6)


def associate_detections(
    detections: np.ndarray,
    tracks: np.ndarray,
    iou_threshold: float = 0.3,
    det_uncertainties: np.ndarray | None = None,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """
    Match detections to tracks using IoU + optional uncertainty weighting.

    Returns:
        matches: list of (detection_idx, track_idx) pairs
        unmatched_dets: list of detection indices
        unmatched_trks: list of track indices
    """
    if len(tracks) == 0:
        return [], list(range(len(detections))), []
    if len(detections) == 0:
        return [], [], list(range(len(tracks)))

    iou_matrix = iou_batch(detections, tracks)

    # weight by uncertainty: lower IoU threshold for uncertain detections
    if det_uncertainties is not None and len(det_uncertainties) > 0:
        uncertainty_weights = 1.0 - 0.3 * det_uncertainties[:, None]
        iou_matrix = iou_matrix * uncertainty_weights

    cost_matrix = 1.0 - iou_matrix

    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    matches = []
    unmatched_dets = list(range(len(detections)))
    unmatched_trks = list(range(len(tracks)))

    for r, c in zip(row_indices, col_indices):
        if iou_matrix[r, c] >= iou_threshold:
            matches.append((r, c))
            if r in unmatched_dets:
                unmatched_dets.remove(r)
            if c in unmatched_trks:
                unmatched_trks.remove(c)

    return matches, unmatched_dets, unmatched_trks


def associate_bytetrack(
    high_dets: np.ndarray,
    low_dets: np.ndarray,
    tracks: np.ndarray,
    high_threshold: float = 0.3,
    low_threshold: float = 0.2,
) -> tuple[list[tuple[int, int]], list[int], list[int], list[tuple[int, int]]]:
    """
    ByteTrack-style two-stage association.

    First associates high-confidence detections, then tries to match
    remaining tracks with low-confidence detections.
    """
    # first association: high confidence
    matches_high, unmatched_dets_high, unmatched_trks = associate_detections(
        high_dets, tracks, iou_threshold=high_threshold
    )

    # second association: low confidence with remaining tracks
    if len(low_dets) > 0 and len(unmatched_trks) > 0:
        remaining_tracks = tracks[unmatched_trks]
        matches_low_local, _, still_unmatched_trk_local = associate_detections(
            low_dets, remaining_tracks, iou_threshold=low_threshold
        )
        # remap indices back
        matches_low = [
            (d_idx, unmatched_trks[t_idx])
            for d_idx, t_idx in matches_low_local
        ]
        unmatched_trks = [unmatched_trks[i] for i in still_unmatched_trk_local]
    else:
        matches_low = []

    return matches_high, unmatched_dets_high, unmatched_trks, matches_low
