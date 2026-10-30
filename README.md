# ARA 7.0 (2026) Pothole Segmentation - Anti Galau Team

Repositori ini berisi keseluruhan *source code* dan solusi dari tim **Anti Galau** untuk kompetisi Data Science ARA 7.0 tahun 2026 yang diselenggarakan oleh Institut Teknologi Sepuluh Nopember. Misi utama dari proyek ini adalah membangun sistem berbasis AI untuk melakukan *semantic segmentation* dalam memetakan area *pothole* (lubang jalan) secara otomatis dan presisi.

## 📌 Tantangan Utama
Berdasarkan hasil *Exploratory Data Analysis (EDA)* yang mendalam pada *dataset*, terdapat beberapa tantangan teknis:
- **Domain Shift (Clarity & Light Gap):** *Test set* 1.62x lebih tajam dan memiliki *kecerahan (brightness)* yang berbeda dari *train set*.
- **Intensity Overlap:** Terdapat tumpang tindih visual yang kuat antara piksel jalan dan piksel lubang jalan.
- **Noise Visual & Skala:** Mayoritas lubang berukuran kecil (*Log Scale*), serta kemunculan *noise* dari ban kendaraan, marka, atau kotoran.

## 🛠️ Arsitektur & Spesifikasi Model
Solusi kami menggunakan arsitektur *deep learning* mutakhir untuk segmentasi:
- **Architecture:** Unet++ (Nested U-Net) melalui modul `segmentation_models_pytorch`.
- **Encoder (Backbone):** EfficientNet-B4.
- **Pre-trained Weights:** ImageNet-1k.
- **Modifikasi Khusus:** Penambahan *Decoder Dropout (0.2)* untuk mengatasi *overfitting* pada detail tekstur.
- **Loss Function:** *Hybrid Loss Engine* (50% Dice Loss + 50% Smoothed BCE Loss). *Label Smoothing* digunakan untuk mengurangi tingkat *overconfidence* model terhadap tekstur aspal yang ambigu.
- **Optimizer:** AdamW dengan *Learning Rate Scheduler* (`ReduceLROnPlateau`).

## 📊 Pipeline Data & Augmentasi
Untuk memastikan ketahanan model (*robustness*), teknik augmentasi berlapis dilakukan dengan *library* `albumentations`:
1. **Geometri Realistis:** Mensimulasikan *dashcam* melalui `Perspective` dan `ShiftScaleRotate`.
2. **Penyelarasan Ketajaman:** `Sharpen` dan `GaussianBlur` untuk melatih model menangani rentang ketajaman *train* dan *test*.
3. **Lighting & Shadow:** Modifikasi dinamis `RandomShadow`, `RandomGamma`, dan `CLAHE` untuk mengatasi *sun glare* atau kondisi gelap.
4. **Texture Independence:** `CoarseDropout` terukur (10-25px) yang memaksa model mengenali tepian jalan (batas lubang) bukan menghafal butiran tekstur.

## ⚙️ Strategi Pelatihan (Training Strategy)
- **Validasi:** *5-Fold Cross Validation* dengan *Deterministic Splitting* untuk *reproducibility*.
- **Akselerasi:** *Automatic Mixed Precision* (AMP) dan utilisasi `Persistent Workers` pada PyTorch DataLoader.
- **Checkpointing:** Menyimpan bobot terbaik (*Best Model*) per fold menggunakan *Early Stopping* yang dipicu oleh metrik F1-Score/Dice Coefficient.
- **Caching:** Dataset pre-resized pada 512x512 ke direktori sementara untuk meminimalisasi proses I/O dan utilitas CPU.

## 🔍 Post-Processing, Inferensi & Error Analysis
Tahapan *post-processing* diaplikasikan untuk menghasilkan masker (*mask*) segmentasi akhir yang bersih beserta analisis evaluasinya:
- **Test Time Augmentation (TTA):** Menggunakan variasi *horizontal flip* saat inferensi.
- **Ensemble Model:** *Mean Ensemble Probabilities* dari 5 bobot Fold.
- **Morphological Refinement:** Operasi *Opening* (pembersihan debu piksel) dan *Closing* (penggabungan blok prediksi yang terputus).
- **Shape & Area Filter (CCL):** Pembersihan secara algoritmik menggunakan `connectedComponentsWithStats` untuk menyisihkan prediksi linear panjang (seperti ban/trotoar berdasarkan limit *circularity*) serta menghapus *false positive* di bawah *threshold area*.
- **Qualitative Visual Audit:** Melakukan *error analysis* pada kasus kegagalan terburuk (*worst samples*, misalnya akibat *sun glare*), kasus rata-rata (*median*), dan prediksi terbaik (*best*) dengan memetakan *Confidence Heatmap* dan *Error Map* (TP/FP/FN).

## 🚀 Cara Penggunaan
Buka dan jalankan seluruh alur (mulai dari EDA, Training, Post-Processing, hingga Inferensi) melalui *Jupyter Notebook* utama:
`DatSciARA7_Anti Galau.ipynb`

Seluruh proses eksperimen dan dependensi *library* dapat langsung dieksekusi dalam konfigurasi lokal berbasis *Jupyter* maupun secara *remote* (misalnya via Kaggle/Google Colab).
