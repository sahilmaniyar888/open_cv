"""
depth_estimator.py — Monocular Depth Estimation with MiDaS
==========================================================

WHAT YOU'LL LEARN HERE:
- Loading models from torch.hub (different from torchvision)
- Why monocular depth is "relative" not "absolute"
- How to extract depth values at specific pixel locations
- Depth map normalization and visualization

KEY CONCEPT: Monocular depth estimation predicts how far each pixel
is from the camera using a SINGLE RGB image. This is an ill-posed
problem (infinitely many 3D scenes can produce the same 2D image),
so the model learns statistical priors from training data.

MiDaS (by Intel ISL) is the gold standard for zero-shot monocular depth.
It outputs RELATIVE depth (ordering of distances), not METRIC depth
(actual meters). For metric depth, you'd use ZoeDepth or Depth Anything.


"""

import torch
import torch.nn.functional as F
import numpy as np
import warnings
from typing import Tuple


class DepthEstimator:
    """
    Monocular depth estimation using MiDaS.

    Produces a dense depth map where each pixel has a relative depth value.
    Higher values = farther from camera.

    Usage:
        estimator = DepthEstimator()
        depth_map = estimator.estimate(image_tensor)
        depth_at_point = estimator.get_depth_at(depth_map, x=100, y=200)
    """

    def __init__(
        self,
        model_type: str = "MiDaS_small",
        device: torch.device = torch.device("cpu"),
    ):
        """
        Initialize MiDaS depth estimator.

        Args:
            model_type: Which MiDaS variant to use:
                - "MiDaS_small"  : Fast, less accurate (good for learning)
                - "DPT_Hybrid"   : Medium speed, good accuracy
                - "DPT_Large"    : Slow, best accuracy
            device: Compute device
        """
        self.device = device
        self.model_type = model_type

        # --------------------------------------------------------
        # Load MiDaS from torch.hub
        # --------------------------------------------------------
        # torch.hub.load downloads from a GitHub repo:
        #   repo: "intel-isl/MiDaS"
        #   model: one of the variants above
        #
        # First run downloads the model weights (~100MB for small).
        print(f"Loading MiDaS depth model ({model_type})...")
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=FutureWarning,
                module=r"timm\.models\.layers",
            )
            self.model = torch.hub.load(
                "intel-isl/MiDaS",
                model_type,
                trust_repo=True,
                verbose=False,
            )
        self.model.to(device)
        self.model.eval()

        # Load the matching transform for this model variant
        # Each MiDaS variant expects different input resolutions
        midas_transforms = torch.hub.load(
            "intel-isl/MiDaS",
            "transforms",
            trust_repo=True,
            verbose=False,
        )

        if model_type == "MiDaS_small":
            self.transform = midas_transforms.small_transform
        elif model_type == "DPT_Hybrid":
            self.transform = midas_transforms.dpt_transform
        else:
            self.transform = midas_transforms.dpt_transform

        print("Depth estimator ready!")

    @torch.no_grad()
    def estimate(self, image_rgb: np.ndarray) -> np.ndarray:
        """
        Estimate depth from an RGB image.

        Args:
            image_rgb: numpy array (H, W, 3), uint8, RGB format

        Returns:
            depth_map: numpy array (H, W), float32
                       Higher values = farther from camera
                       Normalized to [0, 1] range
        """
        original_h, original_w = image_rgb.shape[:2]

        # --------------------------------------------------------
        # Preprocess: MiDaS transform handles resizing + normalization
        # --------------------------------------------------------
        input_tensor = self.transform(image_rgb).to(self.device)

        # --------------------------------------------------------
        # Run inference
        # --------------------------------------------------------
        prediction = self.model(input_tensor)

        # --------------------------------------------------------
        # Post-process: resize back to original resolution
        # --------------------------------------------------------
        # MiDaS outputs at its internal resolution, so we need to
        # resize the depth map back to match our original image.
        #
        # F.interpolate is PyTorch's resizing function.
        # mode="bicubic" gives smooth interpolation (better than "nearest")
        # align_corners=False is the recommended setting.
        prediction = F.interpolate(
            prediction.unsqueeze(1),           # Add channel dim: (1, H, W) → (1, 1, H, W)
            size=(original_h, original_w),     # Target size
            mode="bicubic",
            align_corners=False,
        ).squeeze()                            # Remove extra dims: (1, 1, H, W) → (H, W)

        # Move to CPU and convert to numpy
        depth_map = prediction.cpu().numpy()

        # --------------------------------------------------------
        # Normalize to [0, 1]
        # --------------------------------------------------------
        # MiDaS outputs inverse depth (close = high, far = low).
        # We normalize so 0 = closest, 1 = farthest.
        depth_min = depth_map.min()
        depth_max = depth_map.max()

        if depth_max - depth_min > 0:
            depth_map = (depth_map - depth_min) / (depth_max - depth_min)
        else:
            depth_map = np.zeros_like(depth_map)

        # Invert so higher = farther (more intuitive for robotics)
        depth_map = 1.0 - depth_map

        return depth_map.astype(np.float32)

    def get_depth_at(
        self,
        depth_map: np.ndarray,
        x: int,
        y: int,
        window_size: int = 5,
    ) -> float:
        """
        Get the depth value at a specific pixel location.

        Instead of reading a single pixel (which can be noisy),
        we average a small window around the target point.

        Args:
            depth_map: (H, W) normalized depth map
            x, y: pixel coordinates
            window_size: size of averaging window (5 = 5x5 patch)

        Returns:
            Averaged depth value in [0, 1]
        """
        h, w = depth_map.shape
        half = window_size // 2

        # Clamp to image bounds
        y_min = max(0, y - half)
        y_max = min(h, y + half + 1)
        x_min = max(0, x - half)
        x_max = min(w, x + half + 1)

        # Average the depth in the window
        patch = depth_map[y_min:y_max, x_min:x_max]
        return float(np.mean(patch))

    def classify_depth_zone(self, depth_value: float) -> str:
        """
        Classify depth into zones for robotic planning.

        These thresholds would be calibrated per-robot in production.
        For a tabletop manipulator:
          - "near":  within arm's reach, can grasp immediately
          - "mid":   visible but may need to move closer
          - "far":   background, not actionable

        Args:
            depth_value: normalized depth in [0, 1]

        Returns:
            One of "near", "mid", "far"
        """
        if depth_value < 0.3:
            return "near"
        elif depth_value < 0.6:
            return "mid"
        else:
            return "far"
