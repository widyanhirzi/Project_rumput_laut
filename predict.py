"""
Evaluation & Prediction YOLOv11 Classification - Brown Seaweed
Shows: Accuracy, Precision, Recall, F1-Score per class and averages
Saves results to results/ folder
"""

import torch
from ultralytics import YOLO
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import csv
import sys
import time
from datetime import datetime

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
BASE_DIR     = Path(__file__).resolve().parent
MODEL_PATH   = BASE_DIR / "runs" / "classify" / "Kappaphycus_alvarezii_datasets_yolov11" / "weights" / "best.pt"
DATASET_PATH = Path(r"E:\project_rumput_laut\Kappaphycus_alvarezii_datasets")
RESULTS_DIR  = BASE_DIR / "results"          # main folder for saving results
SPLIT        = "test"       # split to evaluate: "test" | "val" | "train"
IMGSZ        = 224
DEVICE       = 0
BATCH        = 32
# ─────────────────────────────────────────────

CLASSES = ["algae_infection", "healthy", "ice-ice"]


# ─────────────────────────────────────────────
# DATASET EVALUATION (Accuracy, Precision, Recall, F1)
# ─────────────────────────────────────────────

def evaluate_dataset(split: str = SPLIT):
    """
    Run prediction on all images in the specified split,
    then compute complete classification metrics.
    """
    split_dir = DATASET_PATH / split
    if not split_dir.exists():
        print(f"[ERROR] Folder not found: {split_dir}")
        sys.exit(1)

    print("=" * 65)
    print(f"  YOLOv11 Classification Evaluation - Dataset: {split.upper()}")
    print("=" * 65)
    print(f"  Model  : {MODEL_PATH}")
    print(f"  Split  : {split_dir}")
    if torch.cuda.is_available():
        print(f"  GPU    : {torch.cuda.get_device_name(0)}")
    print("=" * 65)

    # Collect all images and ground truth labels
    image_paths = []
    y_true = []

    for class_name in CLASSES:
        class_dir = split_dir / class_name
        if not class_dir.exists():
            print(f"  [WARNING] Class folder not found: {class_dir}")
            continue
        images = (
            list(class_dir.glob("*.jpg")) +
            list(class_dir.glob("*.jpeg")) +
            list(class_dir.glob("*.png"))
        )
        image_paths.extend(images)
        y_true.extend([CLASSES.index(class_name)] * len(images))
        print(f"  {class_name:14s}: {len(images):4d} images")

    print(f"  {'─'*45}")
    print(f"  Total          : {len(image_paths):4d} images\n")

    if not image_paths:
        print("[ERROR] No images found!")
        sys.exit(1)

    # Load model
    model = YOLO(str(MODEL_PATH))

    # Predict all images
    print(f"Running prediction on {len(image_paths)} images...")
    start = time.time()

    results = model.predict(
        source  = [str(p) for p in image_paths],
        imgsz   = IMGSZ,
        device  = DEVICE,
        batch   = BATCH,
        verbose = False,
    )

    elapsed = time.time() - start
    print(f"Completed in {elapsed:.2f}s ({len(image_paths)/elapsed:.1f} images/sec)\n")

    # Collect predictions
    y_pred = []
    for r in results:
        y_pred.append(r.probs.top1)

    # ── Compute Metrics ──
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec  = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1   = f1_score(y_true, y_pred, average="macro", zero_division=0)

    print("=" * 65)
    print("  EVALUATION RESULTS")
    print("=" * 65)
    print(f"  Accuracy          : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Precision (macro) : {prec:.4f}  ({prec*100:.2f}%)")
    print(f"  Recall    (macro) : {rec:.4f}  ({rec*100:.2f}%)")
    print(f"  F1-Score  (macro) : {f1:.4f}  ({f1*100:.2f}%)")
    print("=" * 65)

    # ── Per-Class Report ──
    print("\n  PER-CLASS REPORT:")
    print("-" * 65)
    report = classification_report(
        y_true, y_pred,
        target_names=CLASSES,
        digits=4,
        zero_division=0
    )
    print(report)

    # ── Confusion Matrix (raw counts) ──
    cm = confusion_matrix(y_true, y_pred)
    print("  CONFUSION MATRIX (raw counts):")
    print("-" * 65)
    header = f"{'':15s}" + "".join(f"{c:>15s}" for c in CLASSES)
    print(header)
    for i, row in enumerate(cm):
        row_str = f"  {CLASSES[i]:13s}" + "".join(f"{v:>15d}" for v in row)
        print(row_str)
    print("\n  (Rows = Ground Truth, Columns = Predicted)")

    # ── Normalized Confusion Matrix ──
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    print("\n  NORMALIZED CONFUSION MATRIX (row %):")
    print("-" * 65)
    print(header)
    for i, row in enumerate(cm_norm):
        row_str = f"  {CLASSES[i]:13s}" + "".join(f"{v:>15.4f}" for v in row)
        print(row_str)
    print("\n  (Rows = Ground Truth, Columns = Predicted)")
    print("=" * 65)

    metrics = {
        "accuracy" : acc,
        "precision": prec,
        "recall"   : rec,
        "f1"       : f1,
        "y_true"   : y_true,
        "y_pred"   : y_pred,
        "report"   : report,
        "cm"       : cm,
        "cm_norm"  : cm_norm,
    }

    # Save all results to folder
    save_results(metrics, split)

    return metrics


# ─────────────────────────────────────────────
# SAVE RESULTS TO FOLDER
# ─────────────────────────────────────────────

def save_results(metrics: dict, split: str):
    """Save summary, CSV report, confusion matrix PNGs, and metrics bar chart."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir   = RESULTS_DIR / f"predict_{split}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    acc     = metrics["accuracy"]
    prec    = metrics["precision"]
    rec     = metrics["recall"]
    f1      = metrics["f1"]
    cm      = metrics["cm"]
    cm_norm = metrics["cm_norm"]
    report  = metrics["report"]

    # ── 1. Metrics Summary (TXT) ──
    summary_path = out_dir / "metrics_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"YOLOv11 Classification Evaluation - {split.upper()}\n")
        f.write(f"Timestamp : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model     : {MODEL_PATH}\n")
        f.write(f"Dataset   : {DATASET_PATH / split}\n")
        f.write("=" * 50 + "\n")
        f.write(f"Accuracy          : {acc:.4f}  ({acc*100:.2f}%)\n")
        f.write(f"Precision (macro) : {prec:.4f}  ({prec*100:.2f}%)\n")
        f.write(f"Recall    (macro) : {rec:.4f}  ({rec*100:.2f}%)\n")
        f.write(f"F1-Score  (macro) : {f1:.4f}  ({f1*100:.2f}%)\n")
        f.write("=" * 50 + "\n\n")
        f.write("PER-CLASS REPORT:\n")
        f.write(report)
        f.write("\nCONFUSION MATRIX - Raw Counts (Rows=GT, Cols=Pred):\n")
        header = f"{'':16s}" + "".join(f"{c:>15s}" for c in CLASSES)
        f.write(header + "\n")
        for i, row in enumerate(cm):
            f.write(f"  {CLASSES[i]:14s}" + "".join(f"{v:>15d}" for v in row) + "\n")
        f.write("\nCONFUSION MATRIX - Normalized (Rows=GT, Cols=Pred):\n")
        f.write(header + "\n")
        for i, row in enumerate(cm_norm):
            f.write(f"  {CLASSES[i]:14s}" + "".join(f"{v:>15.4f}" for v in row) + "\n")
    print(f"  [Saved] {summary_path}")

    # ── 2. Per-Class Report (CSV) ──
    csv_path = out_dir / "per_class_report.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Class", "Precision", "Recall", "F1-Score", "Support"])
        from sklearn.metrics import precision_recall_fscore_support
        precs, recs, f1s, sups = precision_recall_fscore_support(
            metrics["y_true"], metrics["y_pred"], zero_division=0
        )
        for i, cls in enumerate(CLASSES):
            writer.writerow([cls, f"{precs[i]:.4f}", f"{recs[i]:.4f}",
                             f"{f1s[i]:.4f}", int(sups[i])])
        writer.writerow([])
        writer.writerow(["macro avg", f"{prec:.4f}", f"{rec:.4f}", f"{f1:.4f}", len(metrics['y_true'])])
        writer.writerow(["accuracy", f"{acc:.4f}", "", "", len(metrics['y_true'])])
    print(f"  [Saved] {csv_path}")

    # ── 3. Confusion Matrix - Raw Counts (PNG) ──
    cm_path = out_dir / "confusion_matrix.png"
    _plot_confusion_matrix(
        cm, CLASSES,
        title=f"Confusion Matrix (Raw Counts) - {split.upper()}",
        save_path=cm_path,
        normalize=False,
    )
    print(f"  [Saved] {cm_path}")

    # ── 4. Confusion Matrix - Normalized (PNG) ──
    cm_norm_path = out_dir / "confusion_matrix_normalized.png"
    _plot_confusion_matrix(
        cm_norm, CLASSES,
        title=f"Confusion Matrix (Normalized) - {split.upper()}",
        save_path=cm_norm_path,
        normalize=True,
    )
    print(f"  [Saved] {cm_norm_path}")

    # ── 5. Metrics Bar Chart (PNG) ──
    bar_path = out_dir / "metrics_chart.png"
    from sklearn.metrics import precision_recall_fscore_support as prf
    precs2, recs2, f1s2, _ = prf(metrics["y_true"], metrics["y_pred"], zero_division=0)
    x      = np.arange(len(CLASSES))
    width  = 0.25
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width, precs2, width, label="Precision", color="#4C72B0")
    ax.bar(x,         recs2,  width, label="Recall",    color="#55A868")
    ax.bar(x + width, f1s2,   width, label="F1-Score",  color="#C44E52")
    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES, fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title(f"Per-Class Metrics - {split.upper()}  |  Accuracy: {acc*100:.2f}%", fontsize=12)
    ax.legend(fontsize=10)
    ax.axhline(y=acc, color="gray", linestyle="--", linewidth=1.2, label=f"Accuracy={acc:.4f}")
    for bar_group in ax.containers:
        ax.bar_label(bar_group, fmt="%.3f", fontsize=8, padding=2)
    fig.tight_layout()
    fig.savefig(bar_path, dpi=150)
    plt.close(fig)
    print(f"  [Saved] {bar_path}")

    print(f"\n  All results saved to: {out_dir}\n")


def _plot_confusion_matrix(matrix, class_names, title, save_path, normalize=False):
    """Helper to plot and save a confusion matrix (raw or normalized)."""
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matrix, interpolation="nearest", cmap="Blues",
                   vmin=0, vmax=(1.0 if normalize else None))
    fig.colorbar(im, ax=ax)
    ax.set(
        xticks=range(len(class_names)),
        yticks=range(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted Label",
        ylabel="True Label",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    thresh = matrix.max() / 2.0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            label = f"{val:.2f}" if normalize else str(int(val))
            ax.text(j, i, label, ha="center", va="center",
                    color="white" if val > thresh else "black", fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ─────────────────────────────────────────────
# SINGLE IMAGE / FOLDER PREDICTION
# ─────────────────────────────────────────────

def predict_single(image_path: str):
    """Predict a single image or a folder of images."""
    model = YOLO(str(MODEL_PATH))
    results = model.predict(
        source  = image_path,
        imgsz   = IMGSZ,
        device  = DEVICE,
        verbose = False,
    )

    print("\n" + "=" * 65)
    for r in results:
        probs      = r.probs
        top1_idx   = probs.top1
        top1_conf  = probs.top1conf.item()
        class_name = CLASSES[top1_idx] if top1_idx < len(CLASSES) else str(top1_idx)

        print(f"  Image      : {Path(r.path).name}")
        print(f"  Prediction : {class_name}")
        print(f"  Confidence : {top1_conf:.4f} ({top1_conf*100:.2f}%)")
        print("  All classes:")
        for i, conf in enumerate(probs.data.tolist()):
            name = CLASSES[i] if i < len(CLASSES) else str(i)
            bar  = "#" * int(conf * 30)
            print(f"    {name:14s}: {conf:.4f}  |{bar}")
        print("-" * 65)

    return results


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    if not MODEL_PATH.exists():
        print(f"[ERROR] Model not found: {MODEL_PATH}")
        print("Make sure training is complete and best.pt is available.")
        sys.exit(1)

    if len(sys.argv) == 1:
        # No arguments → evaluate test dataset
        evaluate_dataset(SPLIT)

    elif sys.argv[1] in ("test", "val", "train"):
        # Evaluate a specific split
        evaluate_dataset(sys.argv[1])

    else:
        # Predict single image / folder
        predict_single(sys.argv[1])
