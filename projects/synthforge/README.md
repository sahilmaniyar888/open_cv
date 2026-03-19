# SynthForge: Synthetic Data Generation & Active Learning Pipeline

Automated data engine for training computer vision models — combines procedural synthetic data generation, auto-labeling, active learning, quality assurance, and dataset versioning.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    SynthForge Pipeline                    │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  [Scene Generator] ──► [Domain Randomization]            │
│        │                        │                        │
│        ▼                        ▼                        │
│  Synthetic Images + Labels (YOLO format)                 │
│        │                                                 │
│        ├──► [Augmentation Pipeline]                      │
│        │         (Albumentations + Copy-Paste)            │
│        │                                                 │
│  [Auto-Labeler] ──► High Conf ──► Dataset               │
│        │             Low Conf ──► Human Review            │
│        │                                                 │
│  [Active Learning] ──► Select most informative samples   │
│        │                                                 │
│  [QA Checks] ──► Validate labels, bbox, class balance    │
│        │                                                 │
│  [Versioning] ──► Snapshot + lineage tracking            │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

## Setup

```bash
cd projects/synthforge
pip install -r requirements.txt
```

## Usage

```bash
# quick demo (generates small dataset, augments, runs QA)
python main.py demo

# generate 200 synthetic images
python main.py generate --n-images 200 --version synthetic_v1

# augment existing dataset (3 copies per image)
python main.py augment --source output/synthetic --n-augmented 3 --pipeline heavy

# auto-label real images
python main.py autolabel --input-dir path/to/images --output-dir path/to/output

# active learning: select 50 most informative samples
python main.py select --image-dir path/to/images --budget 50 --acquisition entropy

# run QA checks
python main.py qa --image-dir data/images --label-dir data/labels --num-classes 5

# list dataset versions
python main.py versions --dataset-dir output/synthetic
```

## Key Components

| Module | Description |
|--------|------------|
| `src/scene_generator.py` | Procedural scene creation with geometric objects |
| `src/domain_randomization.py` | 10-axis domain randomization for sim-to-real |
| `src/augmentation.py` | Albumentations pipelines + copy-paste + depth noise |
| `src/auto_labeler.py` | YOLO-based auto-labeling with confidence routing |
| `src/active_learning.py` | Entropy, margin, coreset acquisition functions |
| `src/qa_checks.py` | Label format, bbox, class balance validation |
| `src/versioning.py` | Dataset snapshots with lineage tracking |
| `src/pipeline.py` | End-to-end pipeline orchestrator |
