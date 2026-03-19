"""Active learning module for efficient annotation budget allocation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.metrics.pairwise import cosine_distances


class AcquisitionFunction:
    """Base class for active learning acquisition functions."""

    def score(self, predictions: list[dict]) -> np.ndarray:
        raise NotImplementedError


class EntropySampling(AcquisitionFunction):
    """Select samples with highest prediction entropy (most uncertain)."""

    def score(self, predictions: list[dict]) -> np.ndarray:
        scores = []
        for pred in predictions:
            confidences = pred.get("scores", [])
            if len(confidences) == 0:
                scores.append(1.0)  # no detections = high uncertainty
                continue

            # entropy of detection confidences
            probs = np.array(confidences)
            probs = np.clip(probs, 1e-10, 1.0)
            entropy = -np.sum(probs * np.log(probs) + (1 - probs) * np.log(1 - probs))
            scores.append(entropy / max(len(probs), 1))

        return np.array(scores)


class MarginSampling(AcquisitionFunction):
    """Select samples where top-1 and top-2 predictions are closest."""

    def score(self, predictions: list[dict]) -> np.ndarray:
        scores = []
        for pred in predictions:
            confidences = sorted(pred.get("scores", []), reverse=True)
            if len(confidences) == 0:
                scores.append(1.0)
            elif len(confidences) == 1:
                scores.append(1.0 - confidences[0])
            else:
                margin = confidences[0] - confidences[1]
                scores.append(1.0 - margin)  # smaller margin = more uncertain

        return np.array(scores)


class DetectionCountVariance(AcquisitionFunction):
    """Select samples where detection count varies across augmentations."""

    def score(self, predictions: list[dict]) -> np.ndarray:
        scores = []
        for pred in predictions:
            counts = pred.get("augmented_counts", [len(pred.get("scores", []))])
            scores.append(np.std(counts) if len(counts) > 1 else 0.5)
        return np.array(scores)


class CoresetSampling:
    """
    Select diverse samples using coreset (greedy k-center) approach.

    Picks samples that maximize minimum distance to already-selected set.
    """

    def select(self, features: np.ndarray, n_select: int,
               seed_indices: list[int] | None = None) -> list[int]:
        n = len(features)
        if n_select >= n:
            return list(range(n))

        selected = list(seed_indices) if seed_indices else [0]
        remaining = set(range(n)) - set(selected)

        # precompute distances
        dists = cosine_distances(features)

        for _ in range(n_select - len(selected)):
            # for each remaining point, find min distance to selected set
            min_dists = np.min(dists[list(remaining)][:, selected], axis=1)
            remaining_list = list(remaining)
            # pick the point with maximum min-distance
            best_idx = remaining_list[np.argmax(min_dists)]
            selected.append(best_idx)
            remaining.remove(best_idx)

        return selected


class ActiveLearner:
    """
    Manages the active learning loop for efficient annotation.

    Selects the most informative samples from an unlabeled pool
    for human annotation, maximizing model improvement per label.
    """

    def __init__(
        self,
        acquisition: str = "entropy",
        budget_per_round: int = 50,
    ):
        self.budget = budget_per_round

        if acquisition == "entropy":
            self.acq_fn = EntropySampling()
        elif acquisition == "margin":
            self.acq_fn = MarginSampling()
        elif acquisition == "detection_variance":
            self.acq_fn = DetectionCountVariance()
        else:
            self.acq_fn = EntropySampling()

        self.rounds: list[dict] = []

    def select_samples(
        self,
        predictions: list[dict],
        image_paths: list[str],
        n_select: int | None = None,
    ) -> list[dict]:
        """
        Select most informative samples for annotation.

        Args:
            predictions: list of prediction dicts per image
            image_paths: corresponding image file paths
            n_select: how many to select (defaults to budget_per_round)

        Returns:
            list of selected sample dicts with path, score, rank
        """
        n = n_select or self.budget
        scores = self.acq_fn.score(predictions)

        # rank by acquisition score (higher = more informative)
        ranked_indices = np.argsort(scores)[::-1]
        selected_indices = ranked_indices[:n]

        selected = []
        for rank, idx in enumerate(selected_indices):
            selected.append({
                "path": image_paths[idx],
                "acquisition_score": float(scores[idx]),
                "rank": rank,
                "n_detections": len(predictions[idx].get("scores", [])),
            })

        self.rounds.append({
            "round": len(self.rounds) + 1,
            "n_selected": len(selected),
            "avg_score": float(np.mean(scores[selected_indices])),
            "min_score": float(np.min(scores[selected_indices])),
            "max_score": float(np.max(scores[selected_indices])),
        })

        return selected

    def get_efficiency_curve(self) -> list[dict]:
        """Get active learning efficiency data across rounds."""
        return self.rounds

    def save_selection(self, selected: list[dict], output_path: str):
        """Save selected sample paths to a file for annotation."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            for s in selected:
                f.write(f"{s['path']}\t{s['acquisition_score']:.4f}\n")

        print(f"Saved {len(selected)} samples to {output_path}")
