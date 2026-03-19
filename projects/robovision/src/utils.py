"""
utils.py — Image Loading, Transforms & Helper Functions
========================================================

WHAT YOU'LL LEARN HERE:
- How PyTorch represents images (C, H, W) vs OpenCV (H, W, C)
- How torchvision.transforms preprocesses images for models
- Converting between NumPy arrays, PIL Images, and PyTorch tensors

KEY CONCEPT: Every deep learning model expects input in a specific format.
Most PyTorch vision models expect:
  - Tensor shape: (batch, channels, height, width) — e.g., (1, 3, 224, 224)
  - Pixel values: normalized to [0, 1] or with ImageNet mean/std
  - Channel order: RGB (not BGR like OpenCV!)
"""

import torch
import numpy as np
import os
from PIL import Image
import cv2
from pathlib import Path
from torchvision import transforms


# ============================================================
# COCO Dataset Class Names
# ============================================================
# Faster R-CNN from torchvision is pretrained on COCO dataset
# which has 91 categories. These are the object classes it can detect.
# In a robotics context, the most useful ones are everyday objects
# that a robot would need to manipulate or avoid.

COCO_CLASSES = [
    "__background__", "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
    "N/A", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    "N/A", "backpack", "umbrella", "N/A", "N/A", "handbag", "tie",
    "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "N/A", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "N/A", "dining table", "N/A", "N/A",
    "toilet", "N/A", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "N/A", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]

# Objects that a robot manipulator would typically interact with
MANIPULABLE_OBJECTS = {
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
    "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "book", "vase", "scissors", "cell phone", "remote", "mouse",
    "keyboard", "laptop", "toothbrush", "teddy bear", "backpack",
}


def load_image(image_path: str) -> np.ndarray:
    """
    Load an image from disk using OpenCV.

    IMPORTANT: OpenCV loads images in BGR format, but PyTorch models
    expect RGB. We convert here so all downstream code works with RGB.

    Args:
        image_path: Path to the image file

    Returns:
        numpy array of shape (H, W, 3) in RGB format, dtype uint8
    """
    # cv2.imread returns BGR by default
    image_bgr = cv2.imread(image_path)

    if image_bgr is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    # Convert BGR → RGB (swap first and last channels)
    # This is one of the most common bugs in CV code!
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    return image_rgb


def image_to_tensor(image: np.ndarray) -> torch.Tensor:
    """
    Convert a NumPy image to a PyTorch tensor suitable for model input.

    The transformation chain:
    1. NumPy (H, W, C) uint8 [0-255] → PIL Image
    2. PIL Image → Tensor (C, H, W) float32 [0.0-1.0]
    3. Add batch dimension → (1, C, H, W)

    WHY PIL? torchvision.transforms.ToTensor() expects a PIL Image
    or numpy array and handles the normalization + transpose automatically.

    Args:
        image: numpy array (H, W, 3) in RGB, uint8

    Returns:
        tensor of shape (1, 3, H, W), float32, values in [0, 1]
    """
    # Step 1: Convert numpy → PIL Image
    pil_image = Image.fromarray(image)

    # Step 2: Define the transform pipeline
    # ToTensor() does two things:
    #   a) Transposes (H, W, C) → (C, H, W)
    #   b) Scales uint8 [0, 255] → float32 [0.0, 1.0]
    transform = transforms.ToTensor()

    # Step 3: Apply transform and add batch dimension
    # unsqueeze(0) adds dimension at position 0: (3, H, W) → (1, 3, H, W)
    tensor = transform(pil_image).unsqueeze(0)

    return tensor


def get_device() -> torch.device:
    """
    Detect the best available compute device.

    Priority: CUDA GPU → MPS (Apple Silicon) → CPU

    In production at Skild AI, you'd always use CUDA GPUs.
    For learning, CPU is fine — inference on a single image is fast.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon MPS")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    return device


def download_sample_image(save_path: str | None = None) -> str:
    """
    Download a sample image for testing if no image is provided.
    Uses a COCO-style scene with multiple objects.
    """
    import urllib.request

    if save_path is None:
        project_root = Path(__file__).resolve().parent.parent
        save_path = str(project_root / "assets" / "sample.jpg")

    save_path = str(Path(save_path).expanduser().resolve())
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if not os.path.exists(save_path):
        # A kitchen scene from COCO — perfect for robotic manipulation
        url = "https://images.unsplash.com/photo-1556909114-f6e7ad7d3136?w=800"
        print(f"Downloading sample image to {save_path}...")
        try:
            urllib.request.urlretrieve(url, save_path)
            print("Download complete!")
        except Exception as e:
            print(f"Could not download image: {e}")
            print(f"Please place your own image at {save_path}")
            return ""

    return save_path


def print_tensor_info(tensor: torch.Tensor, name: str = "Tensor") -> None:
    """
    Debug helper — print key properties of a tensor.
    Use this whenever you're confused about tensor shapes!

    This is the #1 debugging technique in PyTorch:
    When something breaks, print the shape, dtype, and value range.
    """
    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(f"  Shape:  {tensor.shape}")
    print(f"  Dtype:  {tensor.dtype}")
    print(f"  Device: {tensor.device}")
    print(f"  Min:    {tensor.min().item():.4f}")
    print(f"  Max:    {tensor.max().item():.4f}")
    print(f"  Mean:   {tensor.mean().item():.4f}")
    print(f"{'='*50}\n")
