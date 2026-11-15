"""
Training YOLOv11 Classification - Seaweed Disease Dataset
Classes: algae_infection, healthy, ice-ice
"""

import torch
from ultralytics import YOLO
from pathlib import Path
import time

# ─────────────────────────────────────────────
# TRAINING CONFIGURATION
# ─────────────────────────────────────────────
DATASET_PATH = r"E:\project_rumput_laut\mixed_datasets_fix"

# Output directory for training runs (relative to train.py)
BASE_DIR     = Path(__file__).resolve().parent
PROJECT_NAME = str(BASE_DIR / "runs" / "classify")
RUN_NAME     = "mixed_datasets_yolov11_fix"

# YOLOv11 Classification Model
# Options: yolo11n-cls.yaml | yolo11s-cls.yaml | yolo11m-cls.yaml | yolo11l-cls.yaml | yolo11x-cls.yaml
#          (n=nano/fastest, x=extra-large/most accurate)
MODEL        = "yolo11n-cls.yaml"

# Hyperparameters
SEED         = 42           # Random seed for reproducibility
EPOCHS       = 300
IMGSZ        = 224          # Input image size
BATCH        = 32           # Adjust according to available VRAM
LR0          = 0.01         # Initial learning rate
LRF          = 0.01         # Final learning rate (LR0 * LRF)
PATIENCE     = 50           # Early stopping patience (0 = disabled)
WORKERS      = 8            # DataLoader worker processes
DEVICE       = 0            # 0 = primary GPU
OPTIMIZER    = "AdamW"      # SGD | Adam | AdamW | RMSProp
AMP          = True         # Automatic Mixed Precision
PRETRAINED   = True         # Use pretrained backbone weights

# Augmentation Settings
AUGMENT      = True
DEGREES      = 15.0         # Rotation angle (degrees)
FLIPLR       = 0.5          # Horizontal flip probability
FLIPUD       = 0.0          # Vertical flip probability
HSV_H        = 0.015        # Hue augmentation
HSV_S        = 0.7          # Saturation augmentation
HSV_V        = 0.4          # Brightness augmentation

# ─────────────────────────────────────────────


def print_banner():
    print("=" * 60)
    print("  YOLOv11 Classification - Seaweed Disease")
    print("=" * 60)
    print(f"  Model     : {MODEL}")
    print(f"  Dataset   : {DATASET_PATH}")
    print(f"  Epochs    : {EPOCHS}")
    print(f"  Batch     : {BATCH}")
    print(f"  Image size: {IMGSZ}x{IMGSZ}")
    if torch.cuda.is_available():
        print(f"  Device    : GPU - {torch.cuda.get_device_name(0)}")
        print(f"  VRAM      : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print(f"  Device    : CPU")
    print(f"  AMP       : {AMP}")
    print(f"  Optimizer : {OPTIMIZER}")
    print("=" * 60)


def check_dataset():
    """Verify that dataset folders and image files exist."""
    dataset = Path(DATASET_PATH)
    splits = ["train", "val", "test"]
    classes = ["algae_infection", "ice-ice", "healthy"]

    print("\nVerifying Dataset:")
    total = 0
    for split in splits:
        for cls in classes:
            path = dataset / split / cls
            if path.exists():
                count = len(list(path.glob("*.jpg")) + list(path.glob("*.png")) + list(path.glob("*.jpeg")))
                print(f"   {split:5s}/{cls:15s}: {count:4d} images")
                total += count
            else:
                print(f"   WARNING: {split}/{cls} not found!")
    print(f"   {'-'*35}")
    print(f"   Total               : {total:4d} images\n")


def train():
    print_banner()
    check_dataset()

    if not torch.cuda.is_available():
        print("WARNING: GPU not detected, training will run on CPU (very slow)!")
    else:
        print(f"GPU: {torch.cuda.get_device_name(0)} ready for training\n")

    print(f"Loading model architecture: {MODEL}")
    model = YOLO(MODEL)

    print("\nStarting training...\n")
    start_time = time.time()

    results = model.train(
        data        = DATASET_PATH,
        epochs      = EPOCHS,
        imgsz       = IMGSZ,
        batch       = BATCH,
        lr0         = LR0,
        lrf         = LRF,
        patience    = PATIENCE,
        workers     = WORKERS,
        device      = DEVICE,
        optimizer   = OPTIMIZER,
        amp         = AMP,
        pretrained  = PRETRAINED,
        augment     = AUGMENT,
        degrees     = DEGREES,
        fliplr      = FLIPLR,
        flipud      = FLIPUD,
        hsv_h       = HSV_H,
        hsv_s       = HSV_S,
        hsv_v       = HSV_V,
        project     = PROJECT_NAME,
        name        = RUN_NAME,
        exist_ok    = True,
        verbose     = True,
        plots       = True,
        save        = True,
        save_period = 10,
        val         = True,
        task        = "classify",
        seed        = SEED,
    )

    elapsed = time.time() - start_time
    hours, rem = divmod(elapsed, 3600)
    minutes, seconds = divmod(rem, 60)

    print("\n" + "=" * 60)
    print("  Training Complete!")
    print(f"  Total time  : {int(hours)}h {int(minutes)}m {int(seconds)}s")
    print(f"  Best model  : {PROJECT_NAME}/{RUN_NAME}/weights/best.pt")
    print(f"  Run directory: {PROJECT_NAME}/{RUN_NAME}/")
    print("=" * 60)

    return results


def validate():
    """Evaluate best model weights on the test split."""
    best_model_path = Path(PROJECT_NAME) / RUN_NAME / "weights" / "best.pt"
    if not best_model_path.exists():
        print(f"Model checkpoint not found at: {best_model_path}")
        return

    print(f"\nEvaluating on TEST split...")
    model = YOLO(str(best_model_path))
    metrics = model.val(
        data    = DATASET_PATH,
        split   = "test",
        imgsz   = IMGSZ,
        batch   = BATCH,
        device  = DEVICE,
        verbose = True,
    )
    print(f"\nTop-1 Accuracy: {metrics.top1:.4f}")
    print(f"Top-5 Accuracy: {metrics.top5:.4f}")
    return metrics


if __name__ == "__main__":
    results = train()
    validate()
