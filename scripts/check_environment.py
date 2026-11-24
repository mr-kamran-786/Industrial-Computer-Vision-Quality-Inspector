import sys
import torch
import cv2
from ultralytics import YOLO

print("Python:", sys.version)
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
    print("CUDA version used by PyTorch:", torch.version.cuda)

print("OpenCV:", cv2.__version__)

model = YOLO("yolo11n.pt")
print("YOLO model loaded successfully.")