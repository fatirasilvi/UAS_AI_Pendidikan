import json
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# ==========================================
# 1. KONFIGURASI HALAMAN UTAMA
# ==========================================
st.set_page_config(
    page_title="Klasifikasi Perangkat Jaringan",
    layout="wide",
)

# Fungsi untuk memuat model dan label menggunakan caching Streamlit
@st.cache_resource
def load_prediction_resources():
    # Memuat model Keras (.keras)
    model = tf.keras.models.load_model("mobilenetv2_perangkat_jaringan.keras")

    # Memuat file mapping index ke nama kelas asli
    with open("labels.json", "r") as f:
        labels_dict = json.load(f)

    # Memastikan key berbentuk integer agar sinkron dengan output prediksi
    labels_dict = {int(k): v for k, v in labels_dict.items()}
    return model, labels_dict

# Menjalankan fungsi memuat resource dengan penanganan error standar
try:
    model, class_labels = load_prediction_resources()
except Exception as e:
    st.error(
        f"Gagal memuat model atau label. Pastikan file "
        f"'mobilenetv2_perangkat_jaringan.keras' dan 'labels.json' berada di direktori yang sama."
    )
    st.stop()

# ==========================================
# 2. SIDEBAR - INFORMASI MODEL
# ==========================================
with st.sidebar:
    st.markdown("### Informasi Arsitektur")
    st.markdown("---")
    st.write("**Model Dasar:** MobileNetV2 (Transfer Learning)")
    st.write("**Ukuran Gambar:** 150 x 150 Piksel")
    st.write("**Jumlah Kelas:** 3")
    
    st.markdown("---")
    st.write("#### Daftar Kelas:")
    for idx, name in sorted(class_labels.items()):
        st.write(f"- {name}")

# ==========================================
# 3. KONTEN UTAMA
# ==========================================
st.title("Aplikasi Klasifikasi Perangkat Jaringan")
st.markdown(
    "Aplikasi inferensi ini berfungsi untuk mengidentifikasi jenis perangkat infrastruktur jaringan "
    "berdasarkan gambar yang diunggah. Model dapat mengenali tiga kategori perangkat, yaitu: Access Point, Router, dan Switch."
)
st.markdown("---")

# Menggunakan layout 2 kolom (Kiri: Unggah Gambar, Kanan: Hasil Analisis)
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### Input Gambar")
    # File uploader terbatas pada format JPG, JPEG, dan PNG tanpa emotikon
    uploaded_file = st.file_uploader(
        "Unggah file gambar di sini", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(
            image, caption="Preview Gambar", use_container_width=True
        )

with col2:
    st.markdown("### Hasil Analisis")

    # Tombol prediksi dengan tipe standar (flat)
    predict_button = st.button("Jalankan Prediksi")

    if predict_button:
        # Pengecekan input file sebelum proses inferensi dilakukan
        if uploaded_file is None:
            st.warning("Peringatan: Silakan unggah file gambar terlebih dahulu sebelum menekan tombol prediksi.")
        else:
            with st.spinner("Memproses gambar..."):

                # ==========================================
                # 4. PROSES PREPROCESSING (SESUAI NOTEBOOK)
                # ==========================================
                # 1. Resize gambar ke dimensi 150x150 piksel
                img_resized = image.resize((150, 150))

                # 2. Konversi gambar menjadi numpy array
                img_array = np.array(img_resized)

                # Konversi channel RGBA ke RGB jika diperlukan
                if img_array.shape[-1] == 4:
                    img_array = img_array[..., :3]

                # 3. Normalisasi nilai piksel (rescale=1./255)
                img_normalized = img_array / 255.0

                # 4. Penambahan dimensi batch (expand_dims)
                img_batch = np.expand_dims(img_normalized, axis=0)

                # ==========================================
                # 5. INFERENSI PREDIKSI & OUTPUT DATA
                # ==========================================
                # Menjalankan model untuk memprediksi matriks gambar
                predictions = model.predict(img_batch)

                # Menentukan index dengan probabilitas paling tinggi
                predicted_class_idx = np.argmax(predictions[0])
                predicted_class_name = class_labels[predicted_class_idx]
                confidence_score = predictions[0][predicted_class_idx] * 100

                # Menampilkan teks hasil klasifikasi dalam format netral (tanpa alert warna hijau/biru)
                st.markdown(f"#### Hasil Klasifikasi: **{predicted_class_name}**")
                st.write(f"Skor Keyakinan: **{confidence_score:.2f}%**")

                st.markdown("---")
                st.write("Distribusi Probabilitas Kelas:")

                # Menampilkan nilai persentase semua kelas menggunakan progress bar flat abu-abu bawaan
                for idx, name in sorted(class_labels.items()):
                    prob = predictions[0][idx]
                    st.write(f"{name}: {prob*100:.2f}%")
                    st.progress(float(prob))
