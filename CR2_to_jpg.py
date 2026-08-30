import os
import rawpy
from PIL import Image
from pathlib import Path

# --- Konfigurasi Path dan Ukuran ---
# Folder sumber gambar asli (.cr2 dan .jpg campuran)
input_dir = "/home/hirzi_ubuntu/project_rumput_laut/dataset_rumput_raw_hijau"

# Folder hasil akhir (Semua akan menjadi .jpg berukuran 640x640)
output_dir = "/home/hirzi_ubuntu/project_rumput_laut/dataset_rumput_processed_hijau"

# Ukuran target
target_size = (640, 640)

def proses_konversi_dataset():
    input_path = Path(input_dir)
    
    if not input_path.exists():
        print(f"Error: Folder {input_dir} tidak ditemukan.")
        return

    print("="*60)
    print("MEMULAI PROSES KONVERSI DAN RESIZE KE JPG (640x640)")
    print("="*60)
    
    # Menelusuri semua file di folder utama dan sub-foldernya
    for file_path in input_path.rglob('*'):
        if file_path.is_file():
            ext = file_path.suffix.lower()
            
            # Filter hanya untuk file .cr2, .jpg, dan .jpeg
            if ext in ['.cr2', '.jpg', '.jpeg']:
                
                # Mendapatkan path relatif untuk mempertahankan struktur folder (BULU_KUCING, SAKIT, SEHAT)
                relative_path = file_path.relative_to(input_dir)
                
                # MEMASTIKAN OUTPUT SELALU .jpg
                output_file_path = Path(output_dir) / relative_path.with_suffix('.jpg')
                
                # Membuat folder tujuan jika belum ada
                output_file_path.parent.mkdir(parents=True, exist_ok=True)
                
                print(f"Memproses {ext.upper()} -> JPG: {relative_path} ... ", end="")
                
                try:
                    # 1. Jika input adalah file RAW Canon (.cr2)
                    if ext == '.cr2':
                        with rawpy.imread(str(file_path)) as raw:
                            rgb_array = raw.postprocess()
                        img = Image.fromarray(rgb_array)
                    
                    # 2. Jika input sudah berupa file gambar biasa (.jpg/.jpeg)
                    else:
                        img = Image.open(str(file_path))
                        # Mengubah ke RGB untuk memastikan bisa disimpan ke JPG tanpa error warna
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                    
                    # 3. Lakukan resize ke 640x640
                    img_resized = img.resize(target_size, Image.Resampling.LANCZOS)
                    
                    # 4. Simpan hasil akhir sebagai JPEG
                    img_resized.save(output_file_path, 'JPEG', quality=95)
                    print("BERHASIL!")
                    
                except Exception as e:
                    print(f"GAGAL! Error: {e}")

    print("="*60)
    print(f"SELESAI! Semua file output berupa JPG kini ada di:\n{output_dir}")
    print("="*60)

if __name__ == "__main__":
    proses_konversi_dataset()