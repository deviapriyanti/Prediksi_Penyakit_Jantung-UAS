import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model
model = joblib.load('xgb_model_best.pkl')  # Sesuaikan path jika di folder lain

st.title("Prediksi Risiko Penyakit Jantung")

# Input pengguna
age = st.slider("Umur (dalam tahun)", 30, 80, 40)
gender = st.selectbox("Jenis Kelamin", ["Perempuan", "Laki-laki"])
height = st.number_input("Tinggi Badan (cm)", min_value=100, max_value=250, value=165)
weight = st.number_input("Berat Badan (kg)", min_value=30, max_value=200, value=70)
ap_hi = st.number_input("Tekanan Darah Sistolik (ap_hi)", value=120)
ap_lo = st.number_input("Tekanan Darah Diastolik (ap_lo)", value=80)
chol = st.selectbox("Kolesterol", ["Normal", "Tinggi", "Sangat Tinggi"])
gluc = st.selectbox("Glukosa", ["Normal", "Tinggi", "Sangat Tinggi"])
# Baris khusus untuk checkbox horizontal
st.markdown("### Gaya Hidup")
smoke = st.checkbox("Merokok")
alco = st.checkbox("Konsumsi Alkohol")
active = st.checkbox("Aktif secara fisik")

# Konversi input ke format numerik
gender_num = 1 if gender == "Perempuan" else 2
chol_num = {"Normal": 1, "Tinggi": 2, "Sangat Tinggi": 3}[chol]
gluc_num = {"Normal": 1, "Tinggi": 2, "Sangat Tinggi": 3}[gluc]

# Fitur tambahan
bmi = weight / ((height / 100) ** 2)
pulse_pressure = ap_hi - ap_lo
mean_arterial_pressure = (ap_hi + 2 * ap_lo) / 3

# Buat DataFrame untuk prediksi
data = pd.DataFrame([{
    'age_years': age,
    'gender': gender_num,
    'height': height,
    'weight': weight,
    'ap_hi': ap_hi,
    'ap_lo': ap_lo,
    'cholesterol': chol_num,
    'gluc': gluc_num,
    'smoke': int(smoke),
    'alco': int(alco),
    'active': int(active),
    'bmi': bmi,
    'pulse_pressure': pulse_pressure,
    'mean_arterial_pressure': mean_arterial_pressure
}])

# Prediksi
if st.button("Prediksi"):
    hasil = model.predict(data)[0]
    proba = model.predict_proba(data)[0][hasil]
    if hasil == 1:
        st.error(f"Hasil: Risiko Penyakit Jantung ({proba:.2%})")
    else:
        st.success(f"Hasil: Sehat ({proba:.2%})")
