from pathlib import Path
from ultralytics import YOLO
import torch

MODEL_PATH = Path("ml/models/pcb_defect_yolo_v1.pt")
SOURCE_PATH = Path("ml/datasets/industrial_defects/test/images")
OUTPUT_DIR = Path("runs/inference/pcb_defect_test_v1")

def main():
    print("=" * 60)
    print("PCB Defect Detection - Image Inference")
    print("=" * 60)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not SOURCE_PATH.exists():
        raise FileNotFoundError(f"Test image folder not found: {SOURCE_PATH}")

    print(f"Model: {MODEL_PATH}")
    print(f"Source images: {SOURCE_PATH}")
    print("CUDA available:", torch.cuda.is_available())

    model = YOLO(str(MODEL_PATH))

    results = model.predict(
        source=str(SOURCE_PATH),
        imgsz=640,
        conf=0.25,
        save=True,
        save_txt=True,
        save_conf=True,
        project="runs/inference",
        name="pcb_defect_test_v1",
        exist_ok=True,
        device=0 if torch.cuda.is_available() else "cpu"
    )

    print("\nInference complete.")
    print(f"Results saved to: {OUTPUT_DIR}")
    print(f"Total images processed: {len(results)}")

if __name__ == "__main__":
    main()