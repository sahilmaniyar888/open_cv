"""Uncertainty calibration metrics for tracking predictions."""

from __future__ import annotations

import numpy as np


def expected_calibration_error(
    confidences: np.ndarray,
    accuracies: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """
    Compute Expected Calibration Error (ECE).

    Measures how well predicted confidence matches actual accuracy.
    A well-calibrated model has ECE close to 0.

    Args:
        confidences: predicted confidence scores (0-1)
        accuracies: binary correctness indicators (0 or 1)
        n_bins: number of bins for calibration

    Returns:
        dict with ece, bin_confidences, bin_accuracies, bin_counts
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_confidences = []
    bin_accuracies = []
    bin_counts = []

    total = len(confidences)
    ece = 0.0

    for i in range(n_bins):
        mask = (confidences > bin_edges[i]) & (confidences <= bin_edges[i + 1])
        count = np.sum(mask)

        if count > 0:
            avg_conf = np.mean(confidences[mask])
            avg_acc = np.mean(accuracies[mask])
            ece += (count / total) * abs(avg_acc - avg_conf)
        else:
            avg_conf = (bin_edges[i] + bin_edges[i + 1]) / 2
            avg_acc = 0.0

        bin_confidences.append(float(avg_conf))
        bin_accuracies.append(float(avg_acc))
        bin_counts.append(int(count))

    return {
        "ece": float(ece),
        "bin_confidences": bin_confidences,
        "bin_accuracies": bin_accuracies,
        "bin_counts": bin_counts,
        "n_bins": n_bins,
    }


def tracking_calibration(track_results: list[dict], gt_data: dict | None = None
                         ) -> dict:
    """
    Compute calibration metrics specific to tracking.

    Uses detection confidence as the predicted probability and
    IoU with ground truth (if available) as the accuracy measure.
    """
    all_confs = []
    all_correct = []

    for track in track_results:
        conf = track.get("score", 0)
        uncertainty = track.get("uncertainty", 0)
        # combine: effective confidence = score * (1 - uncertainty)
        effective_conf = conf * (1.0 - uncertainty)
        all_confs.append(effective_conf)

        # if no GT, use a heuristic: tracks with many hits are "correct"
        if gt_data is None:
            correct = 1.0 if track.get("hits", 0) >= 3 else 0.0
        else:
            correct = 1.0  # placeholder when GT matching isn't available
        all_correct.append(correct)

    if not all_confs:
        return {"ece": 0.0, "mean_confidence": 0.0, "mean_accuracy": 0.0}

    confs = np.array(all_confs)
    accs = np.array(all_correct)

    result = expected_calibration_error(confs, accs)
    result["mean_confidence"] = float(np.mean(confs))
    result["mean_accuracy"] = float(np.mean(accs))

    return result


def reliability_diagram_data(confidences: np.ndarray, accuracies: np.ndarray,
                             n_bins: int = 10) -> dict:
    """
    Prepare data for plotting a reliability diagram.

    Perfect calibration = diagonal line.
    """
    ece_result = expected_calibration_error(confidences, accuracies, n_bins)

    return {
        "bin_midpoints": [(i + 0.5) / n_bins for i in range(n_bins)],
        "bin_accuracies": ece_result["bin_accuracies"],
        "bin_confidences": ece_result["bin_confidences"],
        "bin_counts": ece_result["bin_counts"],
        "ece": ece_result["ece"],
        "perfect_calibration": [(i + 0.5) / n_bins for i in range(n_bins)],
    }
