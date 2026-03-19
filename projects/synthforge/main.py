#!/usr/bin/env python3
"""SynthForge: Synthetic Data Generation & Active Learning Pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.pipeline import SynthForgePipeline
from src.qa_checks import DatasetQA
from src.versioning import DatasetVersion


PROJECT_ROOT = Path(__file__).resolve().parent


def load_config(config_path: str | None = None) -> dict:
    path = Path(config_path) if config_path else PROJECT_ROOT / "configs" / "default.json"
    with open(path) as f:
        return json.load(f)


def cmd_generate(args, config):
    """Generate synthetic dataset."""
    pipeline = SynthForgePipeline(config)
    result = pipeline.generate_synthetic_dataset(
        n_images=args.n_images,
        version_name=args.version or "synthetic_v1",
    )

    print(f"\nGenerated {result['n_images']} images")
    print(f"Dataset: {result['dataset_dir']}")
    qa = result["qa_results"]
    print(f"QA: {'PASSED' if qa['passed'] else 'FAILED'} "
          f"({qa['errors']} errors, {qa['warnings']} warnings)")


def cmd_augment(args, config):
    """Augment existing dataset."""
    pipeline = SynthForgePipeline(config)
    result = pipeline.augment_dataset(
        source_dir=args.source,
        n_augmented=args.n_augmented,
        pipeline_type=args.pipeline,
    )

    print(f"\nAugmented {result['source_images']} source images")
    print(f"Created {result['augmented_images']} augmented images")
    print(f"Output: {result['output_dir']}")


def cmd_autolabel(args, config):
    """Auto-label images."""
    from src.auto_labeler import AutoLabeler

    al_cfg = config.get("auto_labeler", {})
    labeler = AutoLabeler(
        model_path=al_cfg.get("model", "yolov8n.pt"),
        high_confidence=al_cfg.get("high_confidence", 0.7),
        low_confidence=al_cfg.get("low_confidence", 0.3),
    )

    stats = labeler.label_directory(args.input_dir, args.output_dir)

    print(f"\nAuto-labeled {stats['total_images']} images")
    print(f"  Auto-approved: {stats['total_auto']}")
    print(f"  Needs review:  {stats['total_review']}")
    print(f"  Rejected:      {stats['total_rejected']}")


def cmd_active_learn(args, config):
    """Run active learning selection."""
    pipeline = SynthForgePipeline(config)
    result = pipeline.run_active_learning_selection(
        image_dir=args.image_dir,
        budget=args.budget,
        acquisition=args.acquisition,
    )

    print(f"\nSelected {result['selected']} from {result['total_images']} images")
    print(f"Average acquisition score: {result['avg_score']:.4f}")
    print(f"Selection saved to: {result['output']}")


def cmd_qa(args, config):
    """Run QA checks on a dataset."""
    qa = DatasetQA(args.image_dir, args.label_dir,
                   num_classes=args.num_classes)
    result = qa.run_all_checks()

    print(f"\nQA Results: {'PASSED' if result['passed'] else 'FAILED'}")
    print(f"  Errors:   {result['errors']}")
    print(f"  Warnings: {result['warnings']}")
    print(f"  Info:     {result['info']}")

    if result["details"]:
        print("\nTop issues:")
        for issue in result["details"][:10]:
            print(f"  [{issue['severity']}] {issue['type']}: {issue['message']}")


def cmd_versions(args, config):
    """List or compare dataset versions."""
    versioner = DatasetVersion(args.dataset_dir)

    if args.compare:
        v1, v2 = args.compare.split(",")
        diff = versioner.compare_versions(v1.strip(), v2.strip())
        print(f"\nComparing {diff['v1']} vs {diff['v2']}:")
        print(f"  Added:   {len(diff['added_images'])} images")
        print(f"  Removed: {len(diff['removed_images'])} images")
        print(f"  Common:  {diff['common_images']} images")
    else:
        versions = versioner.list_versions()
        if not versions:
            print("No versions found.")
        else:
            print(f"\n{'Version':<20} {'Created':<25} {'Images':<8} {'Hash'}")
            print("-" * 70)
            for v in versions:
                print(f"{v['version']:<20} {v['created_at'][:19]:<25} "
                      f"{v['num_images']:<8} {v['content_hash']}")


def cmd_demo(args, config):
    """Run a quick demo of the full pipeline."""
    print("Running SynthForge demo pipeline...\n")

    pipeline = SynthForgePipeline(config)

    # step 1: generate small synthetic dataset
    print("Step 1: Generating synthetic data...")
    result = pipeline.generate_synthetic_dataset(
        n_images=20,
        version_name="demo_v1",
    )
    print(f"  Created {result['n_images']} images\n")

    # step 2: augment
    print("Step 2: Augmenting dataset...")
    aug_result = pipeline.augment_dataset(
        source_dir=result["dataset_dir"],
        n_augmented=2,
        pipeline_type="standard",
    )
    print(f"  Created {aug_result['augmented_images']} augmented images\n")

    # step 3: QA
    print("Step 3: Running QA checks...")
    qa = DatasetQA(
        str(Path(result["dataset_dir"]) / "images"),
        str(Path(result["dataset_dir"]) / "labels"),
        num_classes=5,
    )
    qa_result = qa.run_all_checks()
    print(f"  QA: {'PASSED' if qa_result['passed'] else 'FAILED'}")
    print(f"  Issues: {qa_result['total_issues']}\n")

    print("Demo completed successfully!")
    print(f"Output directory: {config.get('output_dir', 'output')}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SynthForge: Synthetic Data Generation & Active Learning Pipeline"
    )
    parser.add_argument("--config", type=str, help="Path to config JSON")

    sub = parser.add_subparsers(dest="command")

    # generate
    gen = sub.add_parser("generate", help="Generate synthetic dataset")
    gen.add_argument("--n-images", type=int, default=100)
    gen.add_argument("--version", type=str)

    # augment
    aug = sub.add_parser("augment", help="Augment existing dataset")
    aug.add_argument("--source", type=str, required=True)
    aug.add_argument("--n-augmented", type=int, default=3)
    aug.add_argument("--pipeline", choices=["standard", "heavy"], default="standard")

    # autolabel
    al = sub.add_parser("autolabel", help="Auto-label images")
    al.add_argument("--input-dir", type=str, required=True)
    al.add_argument("--output-dir", type=str, required=True)

    # active learning
    act = sub.add_parser("select", help="Active learning sample selection")
    act.add_argument("--image-dir", type=str, required=True)
    act.add_argument("--budget", type=int, default=50)
    act.add_argument("--acquisition", choices=["entropy", "margin", "detection_variance"],
                     default="entropy")

    # qa
    qa = sub.add_parser("qa", help="Run QA checks")
    qa.add_argument("--image-dir", type=str, required=True)
    qa.add_argument("--label-dir", type=str, required=True)
    qa.add_argument("--num-classes", type=int, default=5)

    # versions
    ver = sub.add_parser("versions", help="Dataset version management")
    ver.add_argument("--dataset-dir", type=str, required=True)
    ver.add_argument("--compare", type=str, help="Compare two versions: v1,v2")

    # demo
    sub.add_parser("demo", help="Run quick demo")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    config = load_config(args.config)

    print("\n" + "=" * 50)
    print("  SynthForge")
    print("  Synthetic Data & Active Learning Pipeline")
    print("=" * 50)

    commands = {
        "generate": cmd_generate,
        "augment": cmd_augment,
        "autolabel": cmd_autolabel,
        "select": cmd_active_learn,
        "qa": cmd_qa,
        "versions": cmd_versions,
        "demo": cmd_demo,
    }

    if args.command in commands:
        commands[args.command](args, config)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
