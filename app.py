import json
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# ==========================================
# 1. KONFIGURASI HALAMAN UTAMA & INTERFACE
# ==========================================
st.set_page_config(
    page_title="Klasifikasi Perangkat Jaringan",
    page_icon="🌐",
    layout="wide",
)


# Fungsi untuk memuat model dan label secara efisien menggunakan caching Streamlit
@st.cache_resource
def load_prediction_resources():
    # Memuat model Keras terbaik (.keras)
    model = tf.keras.models.load_model("mobilenetv2_perangkat_jaringan.keras")

    # Memuat file mapping index ke nama kelas asli
    with open("labels.json", "r") as f:
        labels_dict = json.load(f)

    # Memastikan key dictionary dalam format integer agar sinkron dengan output model
    labels_dict = {int(k): v for k, v in labels_dict.items()}
    return model, labels_dict


# Menjalankan fungsi memuat resource
try:
    model, class_labels = load_prediction_resources()
except Exception as e:
    st.error(
        f"Gagal memuat model atau label. Pastikan file "
        f"'mobilenetv2_perangkat_jaringan.keras' dan 'labels.json' berada di direktori yang sama. Error: {e}"
    )
    st.stop()

# ==========================================
# 2. SIDEBAR - INFORMASI MODEL ARSITEKTUR
# ==========================================
with st.sidebar:
    st.title("🌐 Info Arsitektur")
    st.markdown("---")
    st.markdown("### Detail Model")
    st.info("**Base Model:** MobileNetV2 (Transfer Learning)")
    st.info("**Ukuran Input:** 150 x 150 Piksel")
    st.info("**Jumlah Kelas:** 3 Perangkat Jaringan")

    st.markdown("### Daftar Kelas Terdeteksi:")
    for idx, name in sorted(class_labels.items()):
        st.write(f"- **{name}**")

    st.markdown("---")
    st.caption("Dikembangkan sebagai aplikasi inferensi berbasis Deep Learning.")

# ==========================================
# 3. KONTEN UTAMA - INPUT & INTERFACE UTAMA
# ==========================================
st.title("🌐 Aplikasi Klasifikasi Perangkat Jaringan")
st.markdown(
    "Selamat datang di aplikasi klasifikasi berbasis kecerdasan buatan. "
    "Unggah gambar komponen infrastruktur jaringan Anda untuk mendeteksi secara otomatis apakah gambar tersebut merupakan sebuah **Access Point**, **Router**, atau **Switch**."
)
st.markdown("---")

# Membuat layout 2 kolom untuk tampilan yang rapi (Kolom Kiri: Unggah & Preview, Kolom Kanan: Hasil)
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📸 Masukan Gambar")
    # File uploader untuk menerima format JPG, JPEG, atau PNG sesuai instruksi
    uploaded_file = st.file_uploader(
        "Pilih file gambar...", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        # Membuka file gambar yang diunggah
        image = Image.open(uploaded_file)
        # Menampilkan gambar dengan lebar penuh di kolomnya
        st.image(
            image, caption="Gambar yang Anda Unggah", use_container_width=True
        )

with col2:
    st.subheader("📊 Hasil Prediksi")

    # Tombol aksi untuk memicu proses klasifikasi
    predict_button = st.button("Jalankan Prediksi", type="primary")

    if predict_button:
        # Pengecekan kondisi apabila pengguna menekan tombol sebelum mengunggah gambar
        if uploaded_file is None:
            st.error("⚠️ Mohon unggah gambar terlebih dahulu!")
        else:
            # Spinner bawaan Streamlit berjalan saat kalkulasi tensor berlangsung
            with st.spinner("Sedang menganalisis gambar, harap tunggu..."):

                # ==========================================
                # 4. PROSES PREPROCESSING (PERSIS SAMA DENGAN TRAINING)
                # ==========================================
                # 1. Resize gambar ke resolusi 150x150 piksel sesuai spesifikasi training
                img_resized = image.resize((150, 150))

                # 2. Konversi objek PIL Image menjadi numpy array
                img_array = np.array(img_resized)

                # Mengatasi masalah jika gambar memiliki channel alpha (RGBA), konversi ke RGB
                if img_array.shape[-1] == 4:
                    img_array = img_array[..., :3]

                # 3. Normalisasi nilai piksel (rescale=1./255) agar bernilai antara 0 - 1
                img_normalized = img_array / 255.0

                # 4. Menambahkan dimensi batch (expand_dims) agar bentuknya menjadi (1, 150, 150, 3)
                img_batch = np.expand_dims(img_normalized, axis=0)

                # ==========================================
                # 5. INFERENSI PREDIKSI & OUTPUT
                # ==========================================
                # Melakukan prediksi dengan model Keras
                predictions = model.predict(img_batch)

                # Mendapatkan index kelas dengan probabilitas tertinggi
                predicted_class_idx = np.argmax(predictions[0])
                # Mengambil nama kelas asli berdasarkan mapping labels.json
                predicted_class_name = class_labels[predicted_class_idx]
                # Menghitung skor keyakinan tertinggi dalam persentase
                confidence_score = predictions[0][predicted_class_idx] * 100

                # Menampilkan metrik utama hasil klasifikasi gambar
                st.success(f"### Kategori Terdeteksi: **{predicted_class_name}**")
                st.metric(
                    label="Skor Keyakinan (Confidence Score)",
                    value=f"{confidence_score:.2f}%",
                )

                st.markdown("---")
                st.write("#### Probabilitas Seluruh Kelas:")

                # Menampilkan bar visualisasi probabilitas untuk masing-masing kelas
                for idx, name in sorted(class_labels.items()):
                    prob = predictions[0][idx]
                    st.write(f"**{name}** ({prob*100:.2f}%)")
                    st.progress(float(prob))