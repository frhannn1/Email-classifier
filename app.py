import streamlit as st
import joblib

# ----------------------------
# Load model dan vectorizer
# ----------------------------
model = joblib.load("model_email_classifier_nb.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# ----------------------------
# UI Streamlit
# ----------------------------
st.set_page_config(page_title="Model Klasifikasi Email", page_icon="📧")
st.title("📧 Klasifikasi Email")

st.write("Masukkan subject dan isi email di bawah ini untuk mengklasifikasikannya:")

# Input dari user
subject = st.text_input("✉️ Subject Email", placeholder="Contoh: Perlu Tindakan Segera")
body = st.text_area("📝 Isi Email", placeholder="Contoh: Mohon segera periksa gangguan jaringan di wilayah Surabaya...")

# Tombol klasifikasi
if st.button("🔍 Klasifikasikan"):
    if subject.strip() == "" and body.strip() == "":
        st.warning("Silakan masukkan subject atau body email terlebih dahulu.")
    else:
        # Gabungkan dan vektorisasi
        full_email = subject + " " + body
        vectorized = vectorizer.transform([full_email])
        prediction = model.predict(vectorized)[0]

        # Tampilkan hasil
        st.success(f"Hasil Klasifikasi: **{prediction}**")

        #  tampilkan probabilitas
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(vectorized)[0]
            st.write("### Probabilitas Klasifikasi:")
            for label, prob in zip(model.classes_, proba):
                st.write(f"- {label}: {prob:.2%}")
