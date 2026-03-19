# Computer Vision & Robotic Perception Portfolio

End-to-end computer vision projects covering 3D perception, multi-object tracking, and synthetic data generation — built for real-world robotic applications.

## Projects

### 1. RoboVision — Depth-Aware 3D Object Detection
> `projects/robovision/`

Real-time perception pipeline fusing **Faster R-CNN** (object detection) with **MiDaS** (monocular depth estimation) to produce structured 3D scene graphs for downstream robot planning.

**Key features:**
- 2D detection + depth fusion into 3D bounding boxes
- Structured scene graph output (JSON) with spatial zones (near/mid/far)
- Supports webcam and image input with configurable depth models

```bash
cd projects/robovision
python main.py --image assets/sample.jpg
python main.py --webcam
```

---

### 2. AdaptTrack — Uncertainty-Aware Multi-Object Tracking
> `projects/adapttrack/`

ByteTrack-style tracker with **Monte Carlo Dropout uncertainty estimation**, Kalman filter state prediction, and built-in failure detection for reliable tracking in dynamic environments.

**Key features:**
- Two-stage detection association (high/low confidence splitting)
- Per-detection uncertainty via MC Dropout or augmented inference
- ReID appearance features (ResNet18) for re-identification
- Failure detection: track fragmentation, ID switches, uncertainty spikes
- MOTA/IDF1/HOTA evaluation metrics with calibration analysis
- Streamlit monitoring dashboard
- Domain adaptation support (DANN, test-time BN adaptation)

```bash
cd projects/adapttrack
python main.py --demo                          # synthetic test
python main.py --webcam --log tracking.jsonl   # live tracking
python main.py --dashboard                     # monitoring UI
```

---

### 3. SynthForge — Synthetic Data Generation & Active Learning
> `projects/synthforge/`

Complete data engine automating the generation, labeling, quality control, and versioning of training data for computer vision models.

**Key features:**
- Procedural synthetic scene generation with 5 geometric object classes
- 10-axis domain randomization (lighting, shadows, blur, motion blur, noise, etc.)
- Albumentations augmentation pipelines (standard + heavy) with copy-paste
- YOLO-based auto-labeling with confidence-based routing (auto-approve / human review)
- Active learning with entropy, margin, and coreset acquisition functions
- Automated QA checks (label format, bbox sanity, class balance)
- Lightweight dataset versioning with lineage tracking

```bash
cd projects/synthforge
python main.py demo                            # quick demo
python main.py generate --n-images 200         # synthetic dataset
python main.py augment --source output/synthetic --pipeline heavy
python main.py select --image-dir images/ --budget 50
python main.py qa --image-dir data/images --label-dir data/labels
```

---

## Learning Scripts
> `scripts/`

Standalone OpenCV and YOLO experiments:

| Script | Description |
|--------|------------|
| `edge_detection.py` | Resize, grayscale, Gaussian blur, Canny edge detection |
| `annotations.py` | Drawing primitives (lines, rectangles, text) on canvas |
| `object_detection.py` | YOLOv8 single-image object detection |
| `live_camera_detection.py` | YOLOv8 real-time webcam detection |
| `motion_detection.py` | Motion detection via background subtraction (MOG2) |

---

## Setup

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/Scripts/activate  # Windows (Git Bash)
# source .venv/bin/activate    # Linux/Mac

# Install all dependencies
pip install -r requirements.txt
```

Each project also has its own `requirements.txt` if you want to install dependencies per-project.

**Note:** Model weights (`.pt`, `.pth`) are not included in the repository. They will be downloaded automatically on first run by ultralytics (YOLOv8) and torch.hub (MiDaS, Faster R-CNN).

---

## Tech Stack

| Category | Tools |
|----------|-------|
| Detection | YOLOv8, Faster R-CNN, RT-DETR |
| Depth | MiDaS (monocular depth estimation) |
| Tracking | Kalman filter, Hungarian algorithm, ByteTrack |
| Uncertainty | MC Dropout, augmented inference |
| Data | Albumentations, OpenCV, synthetic generation |
| ML Framework | PyTorch, torchvision, ultralytics |
| Monitoring | Streamlit |
| Evaluation | MOTA, IDF1, HOTA, ECE calibration |

---

## Repository Structure

```
├── scripts/                    # Standalone OpenCV/YOLO experiments
├── projects/
│   ├── robovision/             # Project 1: 3D perception pipeline
│   │   ├── src/                #   detector, depth, pipeline, scene graph
│   │   └── main.py
│   ├── adapttrack/             # Project 2: Multi-object tracking
│   │   ├── src/                #   tracker, detector, ReID, domain adaptation
│   │   ├── evaluation/         #   MOTA/IDF1/HOTA metrics, calibration
│   │   ├── dashboard/          #   Streamlit monitoring
│   │   └── main.py
│   └── synthforge/             # Project 3: Data generation pipeline
│       ├── src/                #   generator, augmentation, auto-labeler, AL
│       └── main.py
├── requirements.txt            # Consolidated dependencies
└── README.md
```
