"""
detector.py — Object Detection with Faster R-CNN
=================================================

WHAT YOU'LL LEARN HERE:
- Loading a pretrained model from torchvision.models.detection
- Running inference with torch.no_grad() (why it matters)
- Understanding model output format (boxes, labels, scores)
- Filtering detections by confidence threshold
- Non-Maximum Suppression (NMS) — how it works and why it's needed

KEY CONCEPT: Faster R-CNN is a two-stage detector:
  Stage 1 — Region Proposal Network (RPN): "Where might objects be?"
            Generates ~2000 candidate bounding boxes
  Stage 2 — Classification Head: "What is each proposal?"
            Classifies each proposal + refines the bounding box

"""

import torch
import torchvision
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn_v2,
    FasterRCNN_ResNet50_FPN_V2_Weights,
)
from dataclasses import dataclass
from typing import List

from src.utils import COCO_CLASSES, MANIPULABLE_OBJECTS


# ============================================================
# Data class for a single detection
# ============================================================
# Using a dataclass instead of a raw dict makes code cleaner
# and gives you autocomplete in VS Code.

@dataclass
class Detection:
    """A single detected object in an image."""
    class_id: int
    class_name: str
    confidence: float
    bbox: List[float]       # [x1, y1, x2, y2] in pixels
    centroid: List[float]   # [cx, cy] center point
    is_manipulable: bool    # Can a robot grab this?


class ObjectDetector:
    """
    Wraps a pretrained Faster R-CNN model for object detection.

    Usage:
        detector = ObjectDetector(confidence_threshold=0.5)
        detections = detector.detect(image_tensor)
    """

    def __init__(
        self,
        confidence_threshold: float = 0.5,
        device: torch.device = torch.device("cpu"),
    ):
        """
        Initialize the detector.

        Args:
            confidence_threshold: Minimum confidence to keep a detection.
                - 0.5 is a common default (good balance of precision/recall)
                - Higher (0.7+) = fewer but more confident detections
                - Lower (0.3) = more detections but more false positives
            device: torch.device for computation (cpu/cuda/mps)
        """
        self.confidence_threshold = confidence_threshold
        self.device = device

        # --------------------------------------------------------
        # Load pretrained Faster R-CNN with ResNet50-FPN backbone
        # --------------------------------------------------------
        # FPN = Feature Pyramid Network, which detects objects at
        # multiple scales (small, medium, large objects).
        # V2 uses improved training recipe for better accuracy.
        #
        # The weights parameter loads pretrained COCO weights.
        # First time you run this, it downloads ~170MB of weights.
        print("Loading Faster R-CNN model...")
        weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        self.model = fasterrcnn_resnet50_fpn_v2(weights=weights)

        # Move model to device (GPU if available)
        self.model.to(self.device)

        # --------------------------------------------------------
        # Set model to evaluation mode
        # --------------------------------------------------------
        # CRITICAL: .eval() does two things:
        #   1. Disables dropout layers (used during training only)
        #   2. Uses running statistics for BatchNorm (not batch stats)
        # Forgetting .eval() is a common bug that silently degrades
        # inference quality!
        self.model.eval()

        print(f"Detector ready! (threshold={confidence_threshold})")

    @torch.no_grad()
    def detect(self, image_tensor: torch.Tensor) -> List[Detection]:
        """
        Run object detection on an image.

        The @torch.no_grad() decorator is IMPORTANT:
        - It disables gradient computation during inference
        - This saves ~50% memory and speeds up computation
        - Gradients are only needed during training (backpropagation)
        - Always use it for inference!

        Args:
            image_tensor: Tensor of shape (1, 3, H, W), values in [0, 1]

        Returns:
            List of Detection objects, sorted by confidence (highest first)
        """
        # Move input to the same device as the model
        image_tensor = image_tensor.to(self.device)

        # --------------------------------------------------------
        # Run the model
        # --------------------------------------------------------
        # Faster R-CNN expects a list of tensors (one per image in batch)
        # Output is also a list of dicts, one per image:
        #   [{"boxes": Tensor, "labels": Tensor, "scores": Tensor}]
        predictions = self.model(image_tensor)

        # We only have 1 image, so take the first result
        pred = predictions[0]

        # --------------------------------------------------------
        # Extract results
        # --------------------------------------------------------
        # boxes:  shape (N, 4) — N detected boxes, each [x1, y1, x2, y2]
        # labels: shape (N,)   — class index for each box
        # scores: shape (N,)   — confidence for each detection [0.0 to 1.0]
        boxes = pred["boxes"].cpu().numpy()
        labels = pred["labels"].cpu().numpy()
        scores = pred["scores"].cpu().numpy()

        # --------------------------------------------------------
        # Filter by confidence threshold
        # --------------------------------------------------------
        # The model outputs ALL proposals, including very low-confidence
        # ones. We keep only those above our threshold.
        #
        # NMS (Non-Maximum Suppression) is already applied internally
        # by torchvision's Faster R-CNN, so we don't need to do it manually.
        # But understanding NMS is critical:
        #
        # NMS ALGORITHM:
        # 1. Sort detections by confidence (highest first)
        # 2. Take the highest-confidence detection
        # 3. Remove all other detections that overlap with it (IoU > 0.5)
        # 4. Repeat until no detections remain
        #
        # This prevents multiple boxes for the same object.
        detections = []

        for box, label, score in zip(boxes, labels, scores):
            if score < self.confidence_threshold:
                continue  # Skip low-confidence detections

            class_name = COCO_CLASSES[label]
            if class_name == "N/A":
                continue  # Skip invalid classes

            # Compute centroid (center of bounding box)
            cx = (box[0] + box[2]) / 2.0
            cy = (box[1] + box[3]) / 2.0

            detection = Detection(
                class_id=int(label),
                class_name=class_name,
                confidence=float(score),
                bbox=[float(b) for b in box],
                centroid=[float(cx), float(cy)],
                is_manipulable=class_name in MANIPULABLE_OBJECTS,
            )
            detections.append(detection)

        # Sort by confidence (highest first)
        detections.sort(key=lambda d: d.confidence, reverse=True)

        print(f"Detected {len(detections)} objects above {self.confidence_threshold} confidence")
        return detections
