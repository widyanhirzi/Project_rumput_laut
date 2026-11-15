# Seaweed Disease Classification using YOLOv11

Automated disease detection and classification for economically vital red seaweed species (*Kappaphycus alvarezii* and *Kappaphycus striatum*) using state-of-the-art **YOLOv11 Classification**.

---

## 📌 Overview

Seaweed farming is a critical coastal industry, yet productivity is severely affected by pathogens and environmental diseases. This project provides an end-to-end deep learning framework for classifying seaweed health conditions into three primary categories:

- **`algae_infection`** — Epiphytic algae contamination (*Ceramium* sp. / filamentous epiphytes)
- **`healthy`** — Healthy thallus free of observable infection
- **`ice-ice`** — Ice-ice disease characterized by thallus whitening, tissue bleaching, and fragmentation

Models are evaluated independently across three configurations:
1. ***Kappaphycus alvarezii* (Green/Brown Cottonii)**
2. ***Kappaphycus striatum* (Saccol/Elba)**
3. **Combined / Mixed Dataset** (Generalization across both species)

---

## 🔬 Dataset & Splitting Methodology

### Specimen-Level Stratified Split (80 : 10 : 10)
To avoid **data leakage** across splits (where photographs of the exact same seaweed thallus/specimen appear in both training and test sets), the dataset is split strictly at the **specimen level** (`Sample` group), while simultaneously ensuring **stratification per disease class**:

$$\text{Specimens}_{\text{train}} \cap \text{Specimens}_{\text{val}} = \emptyset, \quad \text{Specimens}_{\text{train}} \cap \text{Specimens}_{\text{test}} = \emptyset, \quad \text{Specimens}_{\text{val}} \cap \text{Specimens}_{\text{test}} = \emptyset$$

### Dataset Distribution Summary

#### 1. *Kappaphycus alvarezii* (`Kappaphycus_alvarezii_datasets_fix`)
| Class | Total Images | Total Specimens | Train (80%) | Val (10%) | Test (10%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `algae_infection` | 865 | 150 | 692 (80.0%) | 86 (9.9%) | 87 (10.1%) |
| `healthy` | 810 | 129 | 648 (80.0%) | 81 (10.0%) | 81 (10.0%) |
| `ice-ice` | 287 | 64 | 230 (80.1%) | 29 (10.1%) | 28 (9.8%) |
| **Total** | **1,962** | **341** | **1,570 (80.0%)** | **196 (10.0%)** | **196 (10.0%)** |

#### 2. *Kappaphycus striatum* (`Kappaphycus_striatum_datasets_fix`)
| Class | Total Images | Total Specimens | Train (80%) | Val (10%) | Test (10%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `algae_infection` | 576 | 105 | 461 (80.0%) | 58 (10.1%) | 57 (9.9%) |
| `healthy` | 368 | 74 | 294 (79.9%) | 37 (10.1%) | 37 (10.1%) |
| `ice-ice` | 506 | 77 | 405 (80.0%) | 51 (10.1%) | 50 (9.9%) |
| **Total** | **1,450** | **256** | **1,160 (80.0%)** | **146 (10.1%)** | **144 (9.9%)** |

#### 3. Mixed / Combined Dataset (`mixed_datasets_fix`)
| Class | *Alvarezii* | *Striatum* | Train (80%) | Val (10%) | Test (10%) | Total Images |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `algae_infection` | 865 | 576 | 1,153 (80.0%) | 144 (10.0%) | 144 (10.0%) | **1,441** |
| `healthy` | 810 | 368 | 942 (80.0%) | 118 (10.0%) | 118 (10.0%) | **1,178** |
| `ice-ice` | 287 | 506 | 635 (80.1%) | 80 (10.1%) | 78 (9.8%) | **793** |
| **Total** | **1,962** | **1,450** | **2,730 (80.0%)** | **342 (10.0%)** | **340 (10.0%)** | **3,412** |

---

## 📊 Experimental Evaluation Results

Independent evaluation on unseen **test sets** (zero data leakage):

| Model Dataset Configuration | Accuracy | Precision (macro) | Recall (macro) | F1-Score (macro) | Test Support |
| :--- | :---: | :---: | :---: | :---: | :---: |
| ***Kappaphycus alvarezii*** | **90.82%** | **87.88%** | **84.89%** | **86.08%** | 196 images |
| ***Kappaphycus striatum*** | **88.19%** | **88.61%** | **87.96%** | **87.05%** | 144 images |
| **Mixed (Combined Species)** | **91.76%** | **91.59%** | **89.53%** | **90.28%** | 340 images |

Detailed classification reports, confusion matrices, and metrics graphs are located in:
- `results/predict_test_Kappaphycus_alvarezii_datasets_yolov11_fix/`
- `results/predict_test_Kappaphycus_striatum_datasets_yolov11_fix/`
- `results/predict_test_mixed_datasets_yolov11_fix/`

---

## 📂 Repository Structure

```
project_rumput_laut/
├── runs/
│   └── classify/
│       ├── Kappaphycus_alvarezii_datasets_yolov11_fix/   # Alvarezii training run & weights
│       ├── Kappaphycus_striatum_datasets_yolov11_fix/    # Striatum training run & weights
│       └── mixed_datasets_yolov11_fix/                   # Mixed dataset training run & weights
├── results/
│   ├── predict_test_Kappaphycus_alvarezii_datasets_yolov11_fix/  # Test set metrics & confusion matrix
│   ├── predict_test_Kappaphycus_striatum_datasets_yolov11_fix/   # Test set metrics & confusion matrix
│   └── predict_test_mixed_datasets_yolov11_fix/                  # Test set metrics & confusion matrix
├── detail_dataset_english.csv      # Complete metadata with specimen (Sample) annotations
├── detail_dataset_indonesia.csv    # Indonesian metadata record
├── detail_dataset_split_fix.csv    # Master CSV mapping images to the 80:10:10 fix split
├── split_dataset_specimen.py       # Specimen-level & stratified splitting script
├── create_mixed_dataset.py         # Collision-free dataset merger script
├── train.py                        # YOLOv11 training pipeline
├── predict.py                      # Model evaluation & prediction tool
├── content_features.py             # Feature extraction utility for duplicate detection
├── dataset_matcher.py              # Visual similarity & content matcher
├── generate_dataset_csv.py         # EXIF & metadata extraction script
├── .gitignore                      # Git exclusion rules (datasets excluded)
└── README.md                       # Project documentation
```

---

## 🛠️ Installation & Setup

### Requirements
- Python 3.10+
- PyTorch with CUDA support (recommended: NVIDIA GPU with >= 8GB VRAM)

```bash
git clone https://github.com/widyanhirzi/Project_rumput_laut.git
cd Project_rumput_laut

# Create and activate virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install required dependencies
pip install ultralytics scikit-learn matplotlib numpy pandas Pillow
```

---

## 🚀 Usage Guide

### 1. Dataset Preparation & Splitting
To reproduce the specimen-level 80:10:10 split from processed images:
```bash
# Run specimen-level stratified split
python split_dataset_specimen.py

# Create the merged/mixed dataset with collision-free prefixes
python create_mixed_dataset.py
```

### 2. Model Training
Configure `DATASET_PATH` and `RUN_NAME` in `train.py`, then run:
```bash
python train.py
```

Key training parameters (YOLOv11n-cls):
- Image size: `224x224`
- Batch size: `32`
- Optimizer: `AdamW` (`lr0=0.01`, `lrf=0.01`)
- Epochs: `300` (Patience: `50`)
- Mixed Precision (AMP): Enabled

### 3. Model Evaluation on Test Split
To evaluate the trained model on test/val/train splits:
```bash
# Evaluate on test split (default)
python predict.py

# Evaluate on validation split
python predict.py val

# Evaluate on train split
python predict.py train
```
The script automatically generates:
- `metrics_summary.txt`: Accuracy, macro/weighted Precision, Recall, F1, and raw/normalized confusion matrices.
- `per_class_report.csv`: Detailed metrics per class.
- `confusion_matrix.png` & `confusion_matrix_normalized.png`: Visual confusion matrices.
- `metrics_chart.png`: Bar chart comparing Precision, Recall, and F1 per class.

### 4. Single Image or Folder Inference
```bash
python predict.py path/to/image.jpg
python predict.py path/to/folder/
```

---

## 📄 License & Attribution

This project and associated dataset annotations are maintained for academic research in precision aquaculture and seaweed pathology detection.
For inquiries, please contact the repository maintainer: [widyanhirzi@gmail.com](mailto:widyanhirzi@gmail.com).
