# RoboVision Setup

Use the existing root virtual environment in `E:\CV_opencv\.venv`. Do not create a second `venv` inside `ML\robovision-lite`.

## Windows setup from the repo root

```powershell
cd E:\CV_opencv
.\.venv\Scripts\Activate.ps1
python -m pip install -r ML\robovision-lite\requirements.txt
python ML\robovision-lite\main.py
```

## Run with your own image

```powershell
python ML\robovision-lite\main.py --image path\to\your_image.jpg
```

## Optional output directory

```powershell
python ML\robovision-lite\main.py --output-dir custom_outputs
```

Notes:

- The first run downloads pretrained model weights, so it will take longer.
- If `assets\sample.jpg` is missing, the app downloads a sample image into `ML\robovision-lite\assets\sample.jpg`.
- Results are saved under `ML\robovision-lite\outputs\` by default.
- Webcam mode is available with `python ML\robovision-lite\main.py --webcam`.
