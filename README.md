# Project Rumput Laut - YOLOv11 Seaweed Classification

Klasifikasi penyakit rumput laut menggunakan YOLOv11 Classification.

## Deskripsi

Project ini menggunakan model YOLOv11 untuk mengklasifikasikan kondisi rumput laut (*Kappaphycus alvarezii* dan *Kappaphycus striatum*) ke dalam beberapa kategori:

- `algae_infection` — terinfeksi alga
- `healthy` — sehat
- `ice-ice` — terkena penyakit ice-ice

## Struktur File

```
project_rumput_laut/
├── train.py          # Script training YOLOv11
├── predict.py        # Script evaluasi & prediksi
├── .gitignore
└── README.md
```

## Requirements

```bash
pip install ultralytics scikit-learn matplotlib numpy
```

## Cara Penggunaan

### Training
```bash
python train.py
```

### Evaluasi Dataset (Test/Val/Train)
```bash
python predict.py          # evaluasi split test (default)
python predict.py val      # evaluasi split val
python predict.py train    # evaluasi split train
```

### Prediksi Gambar Tunggal / Folder
```bash
python predict.py path/to/image.jpg
python predict.py path/to/folder/
```

## Output Evaluasi

Hasil disimpan di folder `results/predict_<split>_<timestamp>/`:

| File | Keterangan |
|------|------------|
| `metrics_summary.txt` | Ringkasan Accuracy, Precision, Recall, F1 + kedua confusion matrix |
| `per_class_report.csv` | Metrik per kelas dalam format CSV |
| `confusion_matrix.png` | Confusion matrix (raw counts) |
| `confusion_matrix_normalized.png` | Confusion matrix (normalized per baris) |
| `metrics_chart.png` | Grafik bar Precision / Recall / F1 per kelas |

## Model

Model yang digunakan: **YOLOv11n-cls** (`yolo11n-cls.yaml`)
