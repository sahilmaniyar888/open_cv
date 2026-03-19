#!/usr/bin/env python3
"""AdaptTrack: Multi-Object Tracking with Uncertainty Estimation."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from src.detector import UncertaintyDetector
from src.tracker import AdaptTracker
from src.failure_detection import FailureDetector
from src.visualizer import draw_tracks, draw_stats_overlay


PROJECT_ROOT = Path(__file__).resolve().parent


def load_config(config_path: str | None = None) -> dict:
    path = Path(config_path) if config_path else PROJECT_ROOT / "configs" / "default.json"
    with open(path) as f:
        return json.load(f)


def run_webcam(config: dict, log_path: str | None = None):
    """Run tracker on webcam feed."""
    det_cfg = config["detector"]
    trk_cfg = config["tracker"]

    detector = UncertaintyDetector(
        model_path=det_cfg["model"],
        confidence=det_cfg["confidence"],
        mc_samples=det_cfg["mc_samples"],
    )
    tracker = AdaptTracker(**trk_cfg)
    failure_det = FailureDetector(**config.get("failure_detection", {}))

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: could not open webcam")
        return

    log_file = open(log_path, "w") if log_path else None

    print("\nWebcam started. Press 'q' to quit.\n")

    try:
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            t0 = time.time()
            detections = detector.detect(frame)
            tracks = tracker.update(detections)
            failure_det.check(tracks, frame_idx)
            elapsed = time.time() - t0
            fps = 1.0 / elapsed if elapsed > 0 else 0

            vis = draw_tracks(frame, tracks, show_uncertainty=True)
            vis = draw_stats_overlay(vis, tracker.get_stats(), fps)

            cv2.imshow("AdaptTrack", vis)

            if log_file:
                active = [t for t in tracks if t.get("state", 0) == 1]
                avg_unc = np.mean([t["uncertainty"] for t in active]) if active else 0
                recent = failure_det.get_recent_events(5)
                log_entry = {
                    "frame": frame_idx,
                    "active_tracks": len(active),
                    "avg_uncertainty": float(avg_unc),
                    "fps": float(fps),
                    "failure_events": len(recent),
                    "events": [{"frame": e.frame_idx, "type": e.event_type,
                                "severity": e.severity} for e in recent],
                }
                log_file.write(json.dumps(log_entry) + "\n")

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            frame_idx += 1

    finally:
        cap.release()
        cv2.destroyAllWindows()
        if log_file:
            log_file.close()

    print("\nTracking Summary:")
    stats = tracker.get_stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")

    fail_summary = failure_det.get_summary()
    print(f"\nFailure Events: {fail_summary['total_events']}")
    for etype, count in fail_summary["event_counts"].items():
        print(f"  {etype}: {count}")


def run_video(config: dict, video_path: str, output_path: str | None = None,
              gt_path: str | None = None):
    """Run tracker on a video file."""
    det_cfg = config["detector"]
    trk_cfg = config["tracker"]

    detector = UncertaintyDetector(
        model_path=det_cfg["model"],
        confidence=det_cfg["confidence"],
        mc_samples=det_cfg["mc_samples"],
    )
    tracker = AdaptTracker(**trk_cfg)
    failure_det = FailureDetector(**config.get("failure_detection", {}))

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: could not open video {video_path}")
        return

    writer = None
    if output_path:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, 30, (w, h))

    # optionally load GT for evaluation
    if gt_path:
        from evaluation.harness import EvaluationHarness
        harness = EvaluationHarness(config.get("evaluation", {}))

    frame_idx = 0
    timings = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.time()
        detections = detector.detect(frame)
        tracks = tracker.update(detections)
        failure_det.check(tracks, frame_idx)
        elapsed = time.time() - t0
        timings.append(elapsed)

        fps = 1.0 / elapsed if elapsed > 0 else 0
        vis = draw_tracks(frame, tracks)
        vis = draw_stats_overlay(vis, tracker.get_stats(), fps)

        if writer:
            writer.write(vis)

        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"  processed {frame_idx} frames, avg FPS: {1.0 / np.mean(timings):.1f}")

    cap.release()
    if writer:
        writer.release()

    print(f"\nProcessed {frame_idx} frames")
    print(f"Average FPS: {1.0 / np.mean(timings):.1f}")

    stats = tracker.get_stats()
    print("\nTracker Stats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    fail_summary = failure_det.get_summary()
    print(f"\nFailure Events: {fail_summary['total_events']}")
    for etype, count in fail_summary["event_counts"].items():
        print(f"  {etype}: {count}")


def run_demo(config: dict):
    """Run a synthetic demo to verify the pipeline works."""
    print("Running synthetic tracking demo...")

    tracker = AdaptTracker(**config["tracker"])
    failure_det = FailureDetector(**config.get("failure_detection", {}))

    np.random.seed(42)

    # simulate 3 objects moving across frames
    objects = [
        {"start": np.array([100.0, 100.0, 150.0, 150.0]), "vel": np.array([3.0, 1.0, 3.0, 1.0])},
        {"start": np.array([300.0, 200.0, 370.0, 280.0]), "vel": np.array([-2.0, 1.5, -2.0, 1.5])},
        {"start": np.array([500.0, 50.0, 560.0, 120.0]), "vel": np.array([1.0, 2.0, 1.0, 2.0])},
    ]

    for frame_idx in range(100):
        bboxes = []
        for obj in objects:
            bbox = obj["start"] + obj["vel"] * frame_idx
            noise = np.random.normal(0, 2, 4)
            bbox += noise
            bboxes.append(bbox)

        bboxes = np.array(bboxes)
        scores = np.random.uniform(0.6, 0.95, len(bboxes))
        class_ids = np.array([0, 0, 1])
        uncertainties = np.random.uniform(0.05, 0.4, len(bboxes))

        detections = {
            "bboxes": bboxes,
            "scores": scores,
            "class_ids": class_ids,
            "uncertainties": uncertainties,
        }

        tracks = tracker.update(detections)
        failure_det.check(tracks, frame_idx)

    stats = tracker.get_stats()
    print("\nDemo Results:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    fail_summary = failure_det.get_summary()
    print(f"\nFailure Events: {fail_summary['total_events']}")
    print("Demo completed successfully!")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="AdaptTrack: Uncertainty-aware multi-object tracking"
    )
    parser.add_argument("--webcam", action="store_true", help="Run on webcam")
    parser.add_argument("--video", type=str, help="Path to input video")
    parser.add_argument("--output", type=str, help="Path for output video")
    parser.add_argument("--gt", type=str, help="Path to MOT ground truth file")
    parser.add_argument("--config", type=str, help="Path to config JSON")
    parser.add_argument("--log", type=str, help="Path to save tracking log (JSONL)")
    parser.add_argument("--demo", action="store_true", help="Run synthetic demo")
    parser.add_argument("--dashboard", action="store_true",
                        help="Launch monitoring dashboard")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    config = load_config(args.config)

    print("\n" + "=" * 50)
    print("  AdaptTrack")
    print("  Uncertainty-Aware Multi-Object Tracking")
    print("=" * 50)

    if args.dashboard:
        import subprocess
        import sys
        dashboard_path = str(PROJECT_ROOT / "dashboard" / "app.py")
        subprocess.run([sys.executable, "-m", "streamlit", "run", dashboard_path])
    elif args.demo:
        run_demo(config)
    elif args.webcam:
        run_webcam(config, log_path=args.log)
    elif args.video:
        run_video(config, args.video, output_path=args.output, gt_path=args.gt)
    else:
        print("\nUsage:")
        print("  python main.py --demo          # synthetic test")
        print("  python main.py --webcam        # live webcam tracking")
        print("  python main.py --video <path>  # track objects in video")
        print("  python main.py --dashboard     # launch monitoring UI")


if __name__ == "__main__":
    main()
