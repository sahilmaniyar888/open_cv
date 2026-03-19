"""End-to-end data pipeline orchestrator."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from .scene_generator import SceneGenerator
from .domain_randomization import DomainRandomizer
from .augmentation import get_standard_pipeline, get_heavy_pipeline
from .auto_labeler import AutoLabeler
from .active_learning import ActiveLearner
from .qa_checks import DatasetQA
from .versioning import DatasetVersion


class SynthForgePipeline:
    """
    Orchestrates the full synthetic data pipeline:

    1. Generate synthetic scenes
    2. Apply domain randomization
    3. Augment existing real images
    4. Auto-label unlabeled images
    5. Select samples via active learning
    6. Run QA checks
    7. Version the dataset
    """

    def __init__(self, config: dict):
        self.config = config
        self.output_dir = Path(config.get("output_dir", "output"))

        gen_cfg = config.get("generator", {})
        self.generator = SceneGenerator(
            image_size=tuple(gen_cfg.get("image_size", [640, 480])),
            min_objects=gen_cfg.get("min_objects", 2),
            max_objects=gen_cfg.get("max_objects", 8),
        )
        self.randomizer = DomainRandomizer(config.get("domain_randomization"))

    def generate_synthetic_dataset(self, n_images: int = 100,
                                   version_name: str = "synthetic_v1") -> dict:
        """Generate a full synthetic dataset with QA and versioning."""
        dataset_dir = self.output_dir / "synthetic"
        img_dir = dataset_dir / "images"
        lbl_dir = dataset_dir / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        print(f"Generating {n_images} synthetic images...")

        for i in tqdm(range(n_images), desc="Generating"):
            scene = self.generator.generate(seed=i)
            image = scene["image"]
            labels = scene["labels"]

            # apply domain randomization
            image, labels = self.randomizer.apply(image, labels)

            # save
            cv2.imwrite(str(img_dir / f"{i:06d}.jpg"), image)

            # recalculate YOLO labels after randomization
            h, w = image.shape[:2]
            yolo_lines = []
            for lbl in labels:
                x1, y1, x2, y2 = lbl["bbox"]
                cx = ((x1 + x2) / 2) / w
                cy = ((y1 + y2) / 2) / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                if 0 < bw < 1 and 0 < bh < 1:
                    yolo_lines.append(f"{lbl['class_id']} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

            with open(lbl_dir / f"{i:06d}.txt", "w") as f:
                f.write("\n".join(yolo_lines))

        # save class names
        with open(dataset_dir / "classes.txt", "w") as f:
            for cls_id in sorted(self.generator.OBJECT_CLASSES):
                f.write(self.generator.OBJECT_CLASSES[cls_id] + "\n")

        # run QA
        print("Running QA checks...")
        qa = DatasetQA(str(img_dir), str(lbl_dir),
                       num_classes=len(self.generator.OBJECT_CLASSES))
        qa_results = qa.run_all_checks()

        print(f"QA: {qa_results['total_issues']} issues "
              f"({qa_results['errors']} errors, {qa_results['warnings']} warnings)")

        # version it
        versioner = DatasetVersion(str(dataset_dir))
        manifest = versioner.create_version(
            version_name=version_name,
            description=f"synthetic dataset with {n_images} images",
            generation_params=self.config.get("generator", {}),
            qa_results=qa_results,
        )

        return {
            "dataset_dir": str(dataset_dir),
            "n_images": n_images,
            "qa_results": qa_results,
            "version": version_name,
        }

    def augment_dataset(self, source_dir: str, n_augmented: int = 3,
                        pipeline_type: str = "standard") -> dict:
        """
        Create augmented copies of an existing dataset.

        Args:
            source_dir: path to YOLO-format dataset (images/ + labels/)
            n_augmented: number of augmented copies per image
            pipeline_type: "standard" or "heavy"
        """
        src = Path(source_dir)
        src_images = src / "images"
        src_labels = src / "labels"

        aug_dir = self.output_dir / "augmented"
        aug_images = aug_dir / "images"
        aug_labels = aug_dir / "labels"
        aug_images.mkdir(parents=True, exist_ok=True)
        aug_labels.mkdir(parents=True, exist_ok=True)

        if pipeline_type == "heavy":
            pipeline = get_heavy_pipeline()
        else:
            pipeline = get_standard_pipeline()

        image_files = sorted(src_images.glob("*.jpg")) + sorted(src_images.glob("*.png"))
        total_generated = 0

        for img_path in tqdm(image_files, desc="Augmenting"):
            image = cv2.imread(str(img_path))
            if image is None:
                continue

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w = image.shape[:2]

            # load labels
            lbl_path = src_labels / f"{img_path.stem}.txt"
            bboxes = []
            class_ids = []
            if lbl_path.exists():
                with open(lbl_path) as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            cls_id = int(parts[0])
                            cx, cy, bw, bh = [float(v) for v in parts[1:]]
                            x1 = (cx - bw / 2) * w
                            y1 = (cy - bh / 2) * h
                            x2 = (cx + bw / 2) * w
                            y2 = (cy + bh / 2) * h
                            bboxes.append([x1, y1, x2, y2])
                            class_ids.append(cls_id)

            for aug_idx in range(n_augmented):
                try:
                    if bboxes:
                        result = pipeline(
                            image=image_rgb,
                            bboxes=bboxes,
                            class_ids=class_ids,
                        )
                        aug_img = cv2.cvtColor(result["image"], cv2.COLOR_RGB2BGR)
                        aug_bboxes = result["bboxes"]
                        aug_classes = result["class_ids"]
                    else:
                        result = pipeline(image=image_rgb, bboxes=[], class_ids=[])
                        aug_img = cv2.cvtColor(result["image"], cv2.COLOR_RGB2BGR)
                        aug_bboxes = []
                        aug_classes = []
                except Exception:
                    continue

                name = f"{img_path.stem}_aug{aug_idx}"
                cv2.imwrite(str(aug_images / f"{name}.jpg"), aug_img)

                ah, aw = aug_img.shape[:2]
                with open(aug_labels / f"{name}.txt", "w") as f:
                    for bbox, cls_id in zip(aug_bboxes, aug_classes):
                        x1, y1, x2, y2 = bbox
                        cx = ((x1 + x2) / 2) / aw
                        cy = ((y1 + y2) / 2) / ah
                        bw = (x2 - x1) / aw
                        bh = (y2 - y1) / ah
                        f.write(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

                total_generated += 1

        return {
            "source_dir": source_dir,
            "output_dir": str(aug_dir),
            "source_images": len(image_files),
            "augmented_images": total_generated,
        }

    def run_active_learning_selection(
        self,
        image_dir: str,
        model_path: str = "yolov8n.pt",
        budget: int = 50,
        acquisition: str = "entropy",
    ) -> dict:
        """
        Run active learning to select most informative samples for annotation.
        """
        from ultralytics import YOLO

        model = YOLO(model_path)
        img_dir = Path(image_dir)
        image_files = sorted(
            list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
        )

        if not image_files:
            return {"error": "no images found"}

        print(f"Running inference on {len(image_files)} images...")
        predictions = []
        image_paths = []

        for img_path in tqdm(image_files, desc="Predicting"):
            results = model(str(img_path), verbose=False)
            boxes = results[0].boxes

            pred = {
                "scores": boxes.conf.cpu().numpy().tolist() if len(boxes) > 0 else [],
                "n_detections": len(boxes),
            }
            predictions.append(pred)
            image_paths.append(str(img_path))

        learner = ActiveLearner(acquisition=acquisition, budget_per_round=budget)
        selected = learner.select_samples(predictions, image_paths)

        output_path = self.output_dir / "active_learning_selection.txt"
        learner.save_selection(selected, str(output_path))

        return {
            "total_images": len(image_files),
            "selected": len(selected),
            "avg_score": float(np.mean([s["acquisition_score"] for s in selected])),
            "output": str(output_path),
        }
