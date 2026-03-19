"""
End-to-end perception pipeline.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.depth_estimator import DepthEstimator
from src.detector import ObjectDetector
from src.scene_graph import SceneGraph, build_scene_graph, print_scene_summary
from src.utils import get_device, image_to_tensor, load_image
from src.visualizer import colorize_depth_map, draw_combined_result, draw_detections


@dataclass
class PipelineResult:
    """Complete output from one pipeline run."""

    scene_graph: SceneGraph
    detection_vis: np.ndarray
    depth_vis: np.ndarray
    combined_vis: np.ndarray
    original_image: np.ndarray
    depth_map: np.ndarray
    timings: dict


class PerceptionPipeline:
    """Object detection + depth estimation pipeline."""

    def __init__(
        self,
        confidence_threshold: float = 0.5,
        depth_model: str = "MiDaS_small",
        device: Optional[str] = None,
    ) -> None:
        print("\n" + "=" * 60)
        print("  INITIALIZING PERCEPTION PIPELINE")
        print("=" * 60 + "\n")

        self.device = get_device() if device is None else device
        self.detector = ObjectDetector(
            confidence_threshold=confidence_threshold,
            device=self.device,
        )
        self.depth_estimator = DepthEstimator(
            model_type=depth_model,
            device=self.device,
        )

        print("\n Pipeline ready!\n")

    def run(self, image_path: str) -> PipelineResult:
        timings: dict[str, float] = {}

        t0 = time.time()
        print(f"\n Loading image: {image_path}")
        image_rgb = load_image(image_path)
        image_tensor = image_to_tensor(image_rgb)
        timings["load"] = time.time() - t0
        print(f"  Image size: {image_rgb.shape[1]}x{image_rgb.shape[0]}")

        t0 = time.time()
        print("\n Running object detection...")
        detections = self.detector.detect(image_tensor)
        timings["detection"] = time.time() - t0

        t0 = time.time()
        print(" Running depth estimation...")
        depth_map = self.depth_estimator.estimate(image_rgb)
        timings["depth"] = time.time() - t0

        t0 = time.time()
        print(" Building scene graph...")
        scene_graph = build_scene_graph(
            detections=detections,
            depth_map=depth_map,
            image_shape=image_rgb.shape,
            depth_estimator=self.depth_estimator,
        )
        timings["fusion"] = time.time() - t0

        t0 = time.time()
        detection_vis = draw_detections(image_rgb, detections)
        depth_vis = colorize_depth_map(depth_map)
        combined_vis = draw_combined_result(image_rgb, scene_graph, depth_map)
        timings["visualization"] = time.time() - t0

        timings["total"] = sum(timings.values())

        print_scene_summary(scene_graph)
        self._print_timings(timings)

        return PipelineResult(
            scene_graph=scene_graph,
            detection_vis=detection_vis,
            depth_vis=depth_vis,
            combined_vis=combined_vis,
            original_image=image_rgb,
            depth_map=depth_map,
            timings=timings,
        )

    def _print_timings(self, timings: dict[str, float]) -> None:
        """Print an ASCII-safe timing table."""
        print("\n PIPELINE TIMINGS:")
        print("-" * 40)
        total = timings.get("total", 0.0) or 1e-9
        for stage, duration in timings.items():
            bar = "#" * int(duration / total * 20)
            print(f"  {stage:15s} {duration * 1000:7.1f}ms  {bar}")
        print("-" * 40)
        fps = 1.0 / timings["total"] if timings["total"] > 0 else 0
        print(f"  {'TOTAL':15s} {timings['total'] * 1000:7.1f}ms  ({fps:.1f} FPS)")
        print()
