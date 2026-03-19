"""
scene_graph.py — Structured Scene Representation
=================================================

WHAT YOU'LL LEARN HERE:
- Designing data structures for robot perception output
- Converting model outputs into structured, serializable formats
- Why robots need scene graphs, not raw pixels

KEY CONCEPT: A scene graph is a structured representation of a scene
that captures objects, their properties, and relationships. This is
the interface between perception and planning in any robotic system.


"""

import json
import os
from dataclasses import dataclass, asdict
from typing import List, Optional
from datetime import datetime

from src.detector import Detection


@dataclass
class SceneObject:
    """
    A single object in the scene graph with all perception attributes.

    This is what a robot planner would consume to make decisions like:
    - "Which objects can I reach?" (check spatial_zone == "near")
    - "What should I pick up?" (check is_manipulable == True)
    - "Where exactly is it?" (use bbox and centroid)
    """
    object_id: int
    class_name: str
    confidence: float
    bbox_2d: List[float]          # [x1, y1, x2, y2]
    centroid_2d: List[float]      # [cx, cy]
    estimated_depth: float        # 0.0 (near) to 1.0 (far)
    spatial_zone: str             # "near", "mid", "far"
    is_manipulable: bool          # Can a robot grasp this?
    bbox_area_ratio: float        # bbox area / image area (object size)


@dataclass
class SceneGraph:
    """
    Complete scene representation from one frame of perception.

    Contains:
    - All detected objects with their properties
    - Scene-level metadata (image size, timestamp, model info)
    - Summary statistics useful for high-level planning
    """
    timestamp: str
    image_width: int
    image_height: int
    num_objects: int
    num_manipulable: int
    objects: List[SceneObject]
    detector_model: str = "faster_rcnn_resnet50_fpn_v2"
    depth_model: str = "MiDaS_small"


def build_scene_graph(
    detections: List[Detection],
    depth_map,  # np.ndarray but we avoid importing numpy here
    image_shape: tuple,
    depth_estimator=None,
) -> SceneGraph:
    """
    Build a scene graph by fusing detection results with depth information.

    This is the FUSION step — combining 2D detection with depth estimation
    to create a richer understanding of the scene. In production, you'd
    also fuse with:
    - Point clouds from LiDAR
    - Semantic segmentation masks
    - Object pose estimation (6-DoF)
    - Previous frame tracking IDs

    Args:
        detections: List of Detection objects from the detector
        depth_map: numpy array (H, W) with normalized depth values
        image_shape: (H, W, C) of the original image
        depth_estimator: DepthEstimator instance for depth queries

    Returns:
        SceneGraph with all objects and metadata
    """
    img_h, img_w = image_shape[:2]
    image_area = img_h * img_w

    scene_objects = []

    for idx, det in enumerate(detections):
        # --------------------------------------------------------
        # Get depth at the object's centroid
        # --------------------------------------------------------
        cx, cy = int(det.centroid[0]), int(det.centroid[1])

        if depth_estimator is not None:
            depth_value = depth_estimator.get_depth_at(depth_map, cx, cy)
            spatial_zone = depth_estimator.classify_depth_zone(depth_value)
        else:
            depth_value = 0.0
            spatial_zone = "unknown"

        # --------------------------------------------------------
        # Compute bounding box area ratio
        # --------------------------------------------------------
        # This tells us how much of the image the object occupies.
        # Useful for estimating object size / proximity.
        x1, y1, x2, y2 = det.bbox
        bbox_area = (x2 - x1) * (y2 - y1)
        area_ratio = bbox_area / image_area

        scene_obj = SceneObject(
            object_id=idx,
            class_name=det.class_name,
            confidence=round(det.confidence, 4),
            bbox_2d=[round(b, 1) for b in det.bbox],
            centroid_2d=[round(c, 1) for c in det.centroid],
            estimated_depth=round(depth_value, 4),
            spatial_zone=spatial_zone,
            is_manipulable=det.is_manipulable,
            bbox_area_ratio=round(area_ratio, 4),
        )
        scene_objects.append(scene_obj)

    # Build the complete scene graph
    scene_graph = SceneGraph(
        timestamp=datetime.now().isoformat(),
        image_width=img_w,
        image_height=img_h,
        num_objects=len(scene_objects),
        num_manipulable=sum(1 for o in scene_objects if o.is_manipulable),
        objects=scene_objects,
    )

    return scene_graph


def save_scene_graph(scene_graph: SceneGraph, output_path: str) -> None:
    """Save scene graph as a formatted JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # asdict converts dataclass → dict recursively
    data = asdict(scene_graph)

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Scene graph saved to {output_path}")


def print_scene_summary(scene_graph: SceneGraph) -> None:
    """Print a human-readable summary of the scene."""
    print("\n" + "=" * 60)
    print("  SCENE GRAPH SUMMARY")
    print("=" * 60)
    print(f"  Image size: {scene_graph.image_width} x {scene_graph.image_height}")
    print(f"  Objects detected: {scene_graph.num_objects}")
    print(f"  Manipulable objects: {scene_graph.num_manipulable}")
    print("-" * 60)

    for obj in scene_graph.objects:
        manip_tag = " [GRASPABLE]" if obj.is_manipulable else ""
        print(
            f"  #{obj.object_id:2d} | {obj.class_name:15s} | "
            f"conf={obj.confidence:.2f} | "
            f"depth={obj.estimated_depth:.2f} ({obj.spatial_zone:4s})"
            f"{manip_tag}"
        )

    print("=" * 60 + "\n")
