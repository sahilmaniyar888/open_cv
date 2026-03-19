"""Re-identification feature extractor for appearance-based matching."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision import models


class ReIDExtractor:
    """
    Extracts appearance features from detected object crops.

    Uses a lightweight ResNet18 backbone (pretrained on ImageNet)
    to produce 512-d feature vectors for re-identification.
    """

    def __init__(self, device: str | None = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # use resnet18 as a lightweight feature backbone
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.model = nn.Sequential(*list(backbone.children())[:-1])  # remove FC
        self.model = self.model.to(self.device)
        self.model.eval()

        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((128, 64)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def extract(self, frame: np.ndarray, bboxes: np.ndarray) -> np.ndarray:
        """
        Extract feature vectors for each detected bounding box.

        Args:
            frame: full image (BGR, HWC)
            bboxes: (N, 4) array of [x1, y1, x2, y2]

        Returns:
            (N, 512) feature matrix, L2-normalized
        """
        if len(bboxes) == 0:
            return np.empty((0, 512))

        h, w = frame.shape[:2]
        crops = []

        for bbox in bboxes:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if x2 <= x1 or y2 <= y1:
                crops.append(torch.zeros(3, 128, 64))
                continue

            crop = frame[y1:y2, x1:x2]
            crop_rgb = crop[:, :, ::-1].copy()  # BGR -> RGB
            crops.append(self.transform(crop_rgb))

        batch = torch.stack(crops).to(self.device)
        features = self.model(batch).squeeze(-1).squeeze(-1)

        # L2 normalize
        features = features / (features.norm(dim=1, keepdim=True) + 1e-6)

        return features.cpu().numpy()

    @staticmethod
    def cosine_distance(feat_a: np.ndarray, feat_b: np.ndarray) -> np.ndarray:
        """Compute pairwise cosine distance between two feature matrices."""
        if len(feat_a) == 0 or len(feat_b) == 0:
            return np.empty((len(feat_a), len(feat_b)))
        # cosine similarity
        sim = feat_a @ feat_b.T
        return 1.0 - sim
