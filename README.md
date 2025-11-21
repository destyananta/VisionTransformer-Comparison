# Vision Transformer Comparison

Proyek pembelajaran mendalam yang membandingkan performa tiga arsitektur Vision Transformer pada tugas klasifikasi gambar. Proyek ini adalah bagian dari kuliah Deep Learning Semester 7.

## Daftar Isi

- [Deskripsi Project](#deskripsi-project)
- [Arsitektur Model](#arsitektur-model)
- [Dataset](#dataset)
- [Instalasi](#instalasi)
- [Penggunaan](#penggunaan)
- [Konfigurasi](#konfigurasi)
- [Hasil](#hasil)
- [Struktur Project](#struktur-project)
- [Persyaratan Sistem](#persyaratan-sistem)

## Deskripsi Project

Proyek ini membandingkan performa tiga arsitektur Vision Transformer berbasis PyTorch dalam melakukan klasifikasi gambar:

1. **Vision Transformer (ViT)** - Model standar yang menggunakan patch-based approach
2. **Data-Efficient Image Transformers (DeiT)** - Versi yang dioptimalkan untuk efisiensi data
3. **Swin Transformer** - Model dengan window-based attention mechanism

Setiap model dilatih dan dievaluasi menggunakan dataset yang sama untuk perbandingan yang fair.

## Arsitektur Model

### Vision Transformer (ViT)
- Model: `vit_tiny_patch16_224`
- Ukuran: Tiny (parameter minimal)
- Patch size: 16×16
- Input size: 224×224

### Data-Efficient Image Transformers (DeiT)
- Model: `deit_tiny_patch16_224`
- Ukuran: Tiny (parameter minimal)
- Patch size: 16×16
- Input size: 224×224
- Optimisasi: Knowledge distillation, data augmentation

### Swin Transformer
- Model: `swin_tiny_patch4_window7_224`
- Ukuran: Tiny (parameter minimal)
- Patch size: 4×4
- Window size: 7×7
- Input size: 224×224
- Optimisasi: Window-based self-attention

## Dataset

- **Format**: Dataset image-based dengan CSV annotation
- **Struktur**:
  - `train.csv`: File anotasi training set (kolom: filename, label)
  - `test.csv`: File anotasi test set (kolom: filename, label)
  - `train/`: Direktori berisi gambar training (~1,100+ gambar)
  - `test/`: Direktori berisi gambar testing (~277+ gambar)
  - `jawaban.csv`: File prediksi hasil inference
- **Preprocessing**:
  - Resize ke 224×224
  - Normalisasi: ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  - Data augmentation untuk training: random horizontal flip, random rotation (±15°)

## Instalasi

### Prasyarat

- Python 3.8+
- CUDA 12.1 (untuk GPU support)
- pip

### Setup Environment

1. **Clone atau download project ini**

2. **Buat virtual environment** (opsional tapi disarankan):
```bash
python -m venv venv
.\venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

### Instalasi Manual Dependencies

Jika menggunakan GPU dengan CUDA 12.1:
```bash
pip install torch>=1.12.0 torchvision>=0.13.0 timm>=0.6.12 numpy pandas Pillow matplotlib scikit-learn tqdm psutil notebook
```

## Penggunaan

### Running the Notebook

1. **Aktifkan virtual environment**:
```bash
.\venv\Scripts\activate
```

2. **Jalankan Jupyter notebook**:
```bash
jupyter notebook main.ipynb
```

3. **Eksekusi cell secara berurutan**:
   - Cell 1: Verifikasi GPU dan spesifikasi hardware
   - Cell 2: Import library
   - Cell 3: Konfigurasi path dan parameter
   - Cell 4: Load dataset
   - Cell 5: Data transforms dan dataset class
   - Cell 6: Buat dataloaders
   - Cell 7: Definisi model
   - Cell 8: Training functions
   - Cell 9-11: Train individual models (ViT, DeiT, Swin)
   - Cell 12+: Evaluasi dan hasil

### Struktur Training Pipeline

```
main.ipynb
├── Hardware Check & Configuration
├── Library Imports
├── Data Loading & Preprocessing
├── Dataset & DataLoader Creation
├── Model Definitions
├── Training Functions
├── Model Training (ViT, DeiT, Swin)
└── Evaluation & Results
```

## Konfigurasi

Konfigurasi utama dapat diatur di Cell 3 (`main.ipynb`):

```python
CONFIG = {
    'img_size': 224,           # Ukuran input gambar
    'batch_size': 8,           # Batch size untuk training
    'num_epochs': 10,          # Jumlah epoch training
    'learning_rate': 3e-4,     # Learning rate optimizer
    'weight_decay': 0.01,      # Weight decay untuk regularisasi
    'num_workers': 0,          # Number of workers untuk DataLoader
    'device': 'cuda/cpu'       # Device (otomatis terdeteksi)
}
```

## Hasil

Setelah training selesai, model terbaik disimpan dan hasil evaluasi mencakup:

### Model Terbaik (Disimpan)
- `best_ViT_Tiny.pth`: Best Vision Transformer checkpoint
- `best_DeiT_Tiny.pth`: Best DeiT checkpoint  
- `best_Swin_Tiny.pth`: Best Swin Transformer checkpoint

### Metriks Evaluasi
- **Accuracy**: Overall accuracy pada test set
- **Precision, Recall, F1-Score**: Per-class metrics
- **Confusion Matrix**: Visualisasi prediksi vs ground truth
- **Training History**: Loss dan accuracy curves

### Output Prediksi
- `jawaban.csv`: File CSV berisi prediksi model pada test set
  - Kolom: image_id, predicted_label, confidence

## Persyaratan Sistem

### Minimum Requirements
- **CPU**: Intel i5 / AMD Ryzen 5 atau lebih baik
- **RAM**: 8 GB
- **Storage**: 10 GB (untuk dataset + model + dependencies)

### Recommended Requirements
- **GPU**: NVIDIA CUDA-capable (RTX 3060 atau lebih baik)
- **RAM**: 16 GB
- **VRAM**: 6 GB untuk NVIDIA GPU
- **Storage**: 20 GB

### Software Requirements
- **Python**: 3.8 - 3.11
- **CUDA**: 12.1 (untuk GPU support)
- **cuDNN**: 8.x (untuk GPU support)

## Dependencies

- `torch>=1.12.0` - PyTorch framework
- `torchvision>=0.13.0` - Computer vision utilities
- `timm>=0.6.12` - Pretrained model library
- `numpy>=1.21` - Numerical computing
- `pandas>=1.3` - Data manipulation
- `Pillow>=8.0` - Image processing
- `matplotlib>=3.4` - Visualization
- `scikit-learn>=1.0` - ML utilities
- `tqdm>=4.64` - Progress bars
- `psutil>=5.8` - System utilities
- `notebook>=6.4` - Jupyter notebook

## Notes

- **GPU Support**: Project dapat berjalan pada CPU tapi sangat lambat. Sangat disarankan menggunakan GPU NVIDIA dengan CUDA support
- **Reproducibility**: Seed diatur ke 42 untuk hasil yang reproducible
- **Data Augmentation**: Hanya diterapkan pada training set
- **Pretrained Weights**: Semua model menggunakan pretrained weights dari ImageNet

## Troubleshooting

### CUDA/GPU Issues
```bash
# Verify CUDA installation
python -c "import torch; print(torch.cuda.is_available())"

# If CUDA not available, reinstall PyTorch for CPU:
pip uninstall torch torchvision
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Memory Issues
- Reduce `batch_size` di CONFIG
- Reduce `num_epochs` untuk quick test
- Close other applications

### Dataset Not Found
- Pastikan `train.csv`, `test.csv`, `train/`, dan `test/` ada di direktori project root
- Verifikasi nama file dan path case-sensitivity
