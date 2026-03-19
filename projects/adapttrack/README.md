# AdaptTrack: Uncertainty-Aware Multi-Object Tracking

Real-time multi-object tracker with Monte Carlo Dropout uncertainty estimation, ByteTrack-style association, and built-in failure detection monitoring.

## Architecture

```
Input Frame
    │
    ├──► [YOLOv8 + MC Dropout] ──► Detections + Uncertainty
    │                                      │
    ├──► [Kalman Filter Prediction]        │
    │           │                          │
    └──► [ByteTrack Association] ◄─────────┘
                │
                ├──► Track Management (create/update/delete)
                ├──► Failure Detection (fragmentation, ID switch, loss)
                └──► Visualization + Logging

    [Streamlit Dashboard] ◄── tracking_log.jsonl
```

## Setup

```bash
cd projects/adapttrack
pip install -r requirements.txt
```

## Usage

```bash
# synthetic demo (no camera/video needed)
python main.py --demo

# webcam tracking
python main.py --webcam

# webcam with logging for dashboard
python main.py --webcam --log tracking_log.jsonl

# video file
python main.py --video path/to/video.mp4 --output result.mp4

# launch monitoring dashboard
python main.py --dashboard
```

## Key Components

| Module | Description |
|--------|------------|
| `src/detector.py` | YOLOv8 detector with MC Dropout uncertainty |
| `src/tracker.py` | ByteTrack-style multi-object tracker |
| `src/track.py` | Track state with Kalman filter |
| `src/association.py` | IoU + Hungarian matching |
| `src/failure_detection.py` | Runtime failure monitoring |
| `src/visualizer.py` | OpenCV visualization with uncertainty bars |
| `evaluation/metrics.py` | MOTA, IDF1, HOTA metrics |
| `evaluation/harness.py` | Config-driven evaluation runner |
| `dashboard/app.py` | Streamlit monitoring dashboard |
