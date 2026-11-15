"""
Training YOLOv11 Classification - Dataset Rumput Laut Coklat
Kelas: BULU_KUCING, SAKIT, SEHAT
"""

import torch
from ultralytics import YOLO
from pathlib import Path
import time

# ─────────────────────────────────────────────
# KONFIGURASI TRAINING
# ─────────────────────────────────────────────
DATASET_PATH = r"E:\project_rumput_laut\mixed_datasets_fix"

# Folder penyimpanan hasil training (absolut, mengikuti lokasi train.py)
BASE_DIR     = Path(__file__).resolve().parent
PROJECT_NAME = str(BASE_DIR / "runs" / "classify")
RUN_NAME     = "mixed_datasets_yolov11_fix"

# Model YOLOv11 Classification
# Pilihan: yolo11n-cls.pt | yolo11s-cls.pt | yolo11m-cls.pt | yolo11l-cls.pt | yolo11x-cls.pt
#          (n=nano/tercepat, x=extra-large/terakurat)
MODEL        = "yolo11n-cls.yaml"

# Hyperparameter
SEED         = 42           # random seed untuk reproducibility
EPOCHS       = 300
IMGSZ        = 224          # ukuran input gambar
BATCH        = 32           # sesuaikan dengan VRAM (16GB bisa lebih besar)
LR0          = 0.01         # learning rate awal
LRF          = 0.01         # learning rate akhir (LR0 * LRF)
PATIENCE     = 50           # early stopping (0 = nonaktif)
WORKERS      = 8            # DataLoader workers
DEVICE       = 0            # 0 = GPU pertama (RTX 5060 Ti)
OPTIMIZER    = "AdamW"      # SGD | Adam | AdamW | RMSProp
AMP          = True         # Automatic Mixed Precision (hemat VRAM)
PRETRAINED   = True         # gunakan pretrained weights

# Augmentasi
AUGMENT      = True
DEGREES      = 15.0         # rotasi (deg)
FLIPLR       = 0.5          # flip horizontal
FLIPUD       = 0.0          # flip vertikal
HSV_H        = 0.015        # hue augmentasi
HSV_S        = 0.7          # saturation augmentasi
HSV_V        = 0.4          # brightness augmentasi

# ─────────────────────────────────────────────


def print_banner():
    print("=" * 60)
    print("  YOLOv11 Classification - Rumput Laut")
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
    """Verifikasi dataset tersedia."""
    dataset = Path(DATASET_PATH)
    splits = ["train", "val", "test"]
    classes = ["algae_infection", "ice-ice", "healthy"]

    print("\n Verifikasi Dataset:")
    total = 0
    for split in splits:
        for cls in classes:
            path = dataset / split / cls
            if path.exists():
                count = len(list(path.glob("*.jpg")) + list(path.glob("*.png")) + list(path.glob("*.jpeg")))
                print(f"   {split:5s}/{cls:12s}: {count:4d} gambar")
                total += count
            else:
                print(f"   PERINGATAN: {split}/{cls} tidak ditemukan!")
    print(f"   {'-'*35}")
    print(f"   Total               : {total:4d} gambar\n")


def train():
    print_banner()
    check_dataset()

    if not torch.cuda.is_available():
        print("PERINGATAN: GPU tidak terdeteksi, training menggunakan CPU (sangat lambat)!")
    else:
        print(f"GPU: {torch.cuda.get_device_name(0)} siap digunakan\n")

    print(f"Memuat model: {MODEL}")
    model = YOLO(MODEL)

    print("\nMemulai training...\n")
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
    print("  Training Selesai!")
    print(f"  Waktu total  : {int(hours)}j {int(minutes)}m {int(seconds)}s")
    print(f"  Model terbaik: {PROJECT_NAME}/{RUN_NAME}/weights/best.pt")
    print(f"  Folder hasil : {PROJECT_NAME}/{RUN_NAME}/")
    print("=" * 60)

    return results


def validate():
    """Evaluasi model terbaik pada data test."""
    best_model_path = Path(PROJECT_NAME) / RUN_NAME / "weights" / "best.pt"
    if not best_model_path.exists():
        print(f"Model tidak ditemukan di: {best_model_path}")
        return

    print(f"\nEvaluasi pada data TEST...")
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
