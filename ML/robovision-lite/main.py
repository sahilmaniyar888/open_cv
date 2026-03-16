#!/usr/bin/env python3
"""
RoboVision-Lite entry point.


"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

import cv2

from src.pipeline import PerceptionPipeline
from src.scene_graph import save_scene_graph
from src.utils import download_sample_image
from src.visualizer import save_results


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_IMAGE_PATH = PROJECT_ROOT / "assets" / "sample.jpg"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"


def resolve_input_path(path_str: str) -> Path:
    """Resolve an input path from the shell cwd first, then project root."""
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return path

    cwd_candidate = (Path.cwd() / path).resolve()
    if cwd_candidate.exists():
        return cwd_candidate

    return (PROJECT_ROOT / path).resolve()


def resolve_output_dir(path_str: str) -> Path:
    """Resolve an output directory relative to the project root when needed."""
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def run_on_image(args: argparse.Namespace) -> None:
    """Run the pipeline on a single image."""
    image_path = resolve_input_path(args.image)
    output_dir = resolve_output_dir(args.output_dir)

    if not image_path.exists():
        if image_path.resolve() == DEFAULT_IMAGE_PATH.resolve():
            print(f"Image not found: {image_path}")
            print("Downloading a sample image...")
            downloaded = download_sample_image(str(DEFAULT_IMAGE_PATH))
            if not downloaded:
                print("Please provide an image with --image <path>")
                sys.exit(1)
            image_path = Path(downloaded).resolve()
        else:
            print(f"Image not found: {image_path}")
            print("Please provide a valid image path with --image <path>")
            sys.exit(1)

    pipeline = PerceptionPipeline(
        confidence_threshold=args.confidence,
        depth_model=args.depth_model,
    )

    result = pipeline.run(str(image_path))

    save_results(
        output_dir=str(output_dir),
        original=result.original_image,
        detection_vis=result.detection_vis,
        depth_vis=result.depth_vis,
        combined_vis=result.combined_vis,
    )

    scene_graph_path = output_dir / "scene_graph.json"
    save_scene_graph(result.scene_graph, str(scene_graph_path))

    print(f"\nAll results saved to {output_dir}")
    print(f"  {output_dir / 'detection_result.jpg'}")
    print(f"  {output_dir / 'depth_map.jpg'}")
    print(f"  {output_dir / 'combined_result.jpg'}")
    print(f"  {output_dir / 'result_grid.jpg'}")
    print(f"  {scene_graph_path}")


def run_on_webcam(args: argparse.Namespace) -> None:
    """Run the pipeline on the default webcam."""
    pipeline = PerceptionPipeline(
        confidence_threshold=args.confidence,
        depth_model=args.depth_model,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        sys.exit(1)

    temp_fd, temp_path = tempfile.mkstemp(prefix="robovision_frame_", suffix=".jpg")
    os.close(temp_fd)

    print("\nWebcam started. Press 'q' to quit.\n")

    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            if not cv2.imwrite(temp_path, frame_bgr):
                print(f"Error: Could not write temporary frame to {temp_path}")
                break

            result = pipeline.run(temp_path)
            display = cv2.cvtColor(result.combined_vis, cv2.COLOR_RGB2BGR)

            fps = 1.0 / result.timings["total"] if result.timings["total"] > 0 else 0
            cv2.putText(
                display,
                f"FPS: {fps:.1f}",
                (10, display.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

            cv2.imshow("RoboVision-Lite", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        cap.release()
        cv2.destroyAllWindows()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="RoboVision-Lite: Depth-aware object detection for robotic perception",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=str(DEFAULT_IMAGE_PATH),
        help="Path to the input image. Defaults to ML/robovision-lite/assets/sample.jpg",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory for saved outputs. Relative paths are created under ML/robovision-lite",
    )
    parser.add_argument(
        "--webcam",
        action="store_true",
        help="Run on webcam instead of a single image",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Detection confidence threshold",
    )
    parser.add_argument(
        "--depth-model",
        type=str,
        default="MiDaS_small",
        choices=["MiDaS_small", "DPT_Hybrid", "DPT_Large"],
        help="MiDaS model variant",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("    RoboVision-Lite")
    print("    Depth-Aware Object Detection for Robotic Perception")
    print("=" * 60)

    if args.webcam:
        run_on_webcam(args)
    else:
        run_on_image(args)


if __name__ == "__main__":
    main()
