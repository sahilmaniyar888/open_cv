# 🤖 RoboVision-Lite: Depth-Aware Object Detection for Robotic Perception



---

## Problem Statement

A robot navigating a real-world environment needs to answer three questions about every object it sees:

1. **What is it?** (object class + confidence)
2. **Where is it in the image?** (2D bounding box)
3. **How far away is it?** (depth / distance estimation)

This project builds a **minimal robotic perception pipeline** that combines:
- A pretrained **object detection** model (Faster R-CNN) for questions 1 & 2
- A pretrained **monocular depth estimation** model (MiDaS) for question 3
- A **scene graph generator** that produces structured JSON output — the kind of data a robot motion planner would consume

This mirrors the hierarchical architecture:  
**Perception → Structured Scene Representation → Action Policy**

---



| Concept | Where in Code |
|---------|--------------|
| PyTorch tensor operations | `src/utils.py` |
| Loading pretrained models from torchvision | `src/detector.py` |
| Loading models from torch.hub | `src/depth_estimator.py` |
| Image preprocessing & transforms | `src/utils.py` |
| Model inference (no training yet) | `src/detector.py`, `src/depth_estimator.py` |
| Non-Maximum Suppression (NMS) | `src/detector.py` |
| Combining multiple model outputs | `src/pipeline.py` |
| Structured scene representation | `src/scene_graph.py` |
| OpenCV visualization | `src/visualizer.py` |
| End-to-end pipeline design | `src/pipeline.py` |

---

## Architecture

```
Input Image
    │
    ├──► [Faster R-CNN]  ──► 2D Detections (class, bbox, confidence)
    │                              │
    ├──► [MiDaS Depth]   ──► Dense Depth Map
    │                              │
    └──► [Fusion Module]  ◄────────┘
              │
              ▼
        Scene Graph (JSON)
        ├── object_id
        ├── class_name
        ├── confidence
        ├── bbox_2d [x1, y1, x2, y2]
        ├── estimated_depth (meters)
        ├── spatial_zone (near/mid/far)
        └── centroid [cx, cy]
```

---

## Project Structure

```
robovision-lite/
├── README.md
├── requirements.txt
├── main.py                    # Entry point — run this
├── src/
│   ├── __init__.py
│   ├── detector.py            # Object detection with Faster R-CNN
│   ├── depth_estimator.py     # Monocular depth with MiDaS
│   ├── pipeline.py            # Combines detection + depth
│   ├── scene_graph.py         # Structured scene output
│   ├── visualizer.py          # OpenCV visualization
│   └── utils.py               # Image loading, transforms, helpers
├── assets/                    # Put your test images here
│   └── sample.jpg
└── outputs/                   # Results saved here
    ├── detection_result.jpg
    ├── depth_map.jpg
    ├── combined_result.jpg
    └── scene_graph.json
```

---

## Setup & Run

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/robovision-lite.git
cd robovision-lite

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add a test image
# Place any image as assets/sample.jpg
# Or use the built-in download in main.py

# 5. Run the pipeline
python main.py --image assets/sample.jpg

# 6. Run on webcam (optional)
python main.py --webcam
```

---

## Key Concepts Explained

### Why Faster R-CNN?
Skild AI's perception stack needs to detect objects for manipulation. Faster R-CNN is a
two-stage detector (Region Proposal Network → Classification) that's still used in robotics
for its accuracy. Understanding it is foundational before moving to DETR or RT-DETR.

### Why Monocular Depth?
Robots with RGB cameras (no LiDAR) need depth from a single image. MiDaS produces
relative depth maps that tell us which objects are closer or farther. In production, this
would be replaced by stereo cameras or RGB-D sensors — but the fusion logic stays the same.

### Why Scene Graphs?
A robot planner doesn't consume raw pixels — it needs structured data. The scene graph
is a list of objects with their properties and spatial relationships. This is exactly what
Skild AI's high-level policy consumes from the perception module.

---

## Exercises After Completing This Project

1. **Add tracking**: Run on a video and maintain object IDs across frames using IoU matching
2. **Add spatial relationships**: Detect "object A is on top of object B" from depth + bbox
3. **Benchmark it**: Measure inference FPS and mAP on COCO validation set
4. **Export to ONNX**: Convert both models to ONNX format for deployment
5. **Add segmentation**: Replace Faster R-CNN with Mask R-CNN for instance masks

---

## License

MIT — Built as a learning project for robotics perception.
