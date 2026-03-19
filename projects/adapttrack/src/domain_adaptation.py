"""Domain adaptation for sim-to-real tracking transfer."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class GradientReversalFunction(torch.autograd.Function):
    """Gradient reversal layer for adversarial domain adaptation (DANN)."""

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


class DomainClassifier(nn.Module):
    """
    Binary domain classifier for adversarial adaptation.

    Classifies features as source (sim) or target (real) domain.
    With gradient reversal, training this makes the feature extractor
    produce domain-invariant representations.
    """

    def __init__(self, feature_dim: int = 512, hidden_dim: int = 256):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, features: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        reversed_features = GradientReversalFunction.apply(features, alpha)
        return self.classifier(reversed_features)


class TestTimeAdaptation:
    """
    Simple test-time adaptation via running statistics update.

    Updates batch normalization statistics on target domain data
    to reduce distribution shift without retraining.
    """

    def __init__(self, model: nn.Module, momentum: float = 0.1):
        self.model = model
        self.momentum = momentum

    def adapt(self, target_features: torch.Tensor, n_steps: int = 10):
        """Update BN statistics using target domain features."""
        self.model.train()

        for module in self.model.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                module.momentum = self.momentum

        with torch.no_grad():
            for _ in range(n_steps):
                _ = self.model(target_features)

        self.model.eval()


def compute_domain_shift(source_features: np.ndarray,
                         target_features: np.ndarray) -> dict:
    """
    Estimate domain shift between source and target feature distributions.

    Uses simple statistics: mean/std shift, MMD approximation.
    """
    src_mean = np.mean(source_features, axis=0)
    tgt_mean = np.mean(target_features, axis=0)
    src_std = np.std(source_features, axis=0)
    tgt_std = np.std(target_features, axis=0)

    # mean shift (L2 distance between distribution centers)
    mean_shift = float(np.linalg.norm(src_mean - tgt_mean))

    # std ratio (how different the spreads are)
    std_ratio = float(np.mean(np.abs(src_std - tgt_std) / (src_std + 1e-6)))

    # simple MMD approximation using mean embeddings
    mmd = mean_shift  # simplified — full MMD uses kernel trick

    return {
        "mean_shift": mean_shift,
        "std_ratio": std_ratio,
        "mmd_approx": mmd,
        "shift_level": "low" if mean_shift < 1.0 else "medium" if mean_shift < 3.0 else "high",
    }
