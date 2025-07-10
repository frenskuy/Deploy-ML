import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Konfigurasi awal halaman
st.set_page_config(page_title="Dashboard Kualitas Udara Dunia", layout="wide")

# Sidebar Navigasi
st.sidebar.title("📌 Navigasi")
menu = st.sidebar.radio("Pilih Halaman:", [
    "Beranda",
    "Eksplorasi Data",
    "Visualisasi Data",
    "Model SVC"
])

# Load Data
DATA_PATH = "AQI and Lat Long of Countries.csv"
data = pd.read_csv(DATA_PATH)
data.dropna(inplace=True)

# ================== BERANDA ==================
if menu == "Beranda":
    st.title("🌍 Dashboard Kualitas Udara Dunia")
    st.markdown("""
    Selamat datang di dashboard interaktif untuk menganalisis kualitas udara dari berbagai negara di dunia berdasarkan data AQI.

    **Fitur:**
    - Eksplorasi data dan statistik dasar
    - Visualisasi distribusi AQI dan polutan
    - Sebaran negara berdasarkan kategori AQI
    - Model klasifikasi kategori AQI menggunakan SVC (Support Vector Classifier)

    **Dataset**: AQI and Lat Long of Countries
    """)
    st.success("Gunakan sidebar untuk menavigasi antara halaman.")

# ================== EKSPLORASI DATA ==================
elif menu == "Eksplorasi Data":
    st.title("🔍 Eksplorasi Data")

    st.subheader("📋 Tabel Data Awal")
    st.dataframe(data.head())

    st.subheader("📊 Statistik Deskriptif")
    st.write(data.describe(include="all"))

    st.subheader("🧼 Data Kosong per Kolom")
    st.write(data.isnull().sum())

    st.subheader("🏙️ 20 Kota dengan AQI Rata-rata Tertinggi")
    top_cities = data.groupby('City')['AQI Value'].mean().sort_values(ascending=False).head(20)
    st.dataframe(top_cities.reset_index())

# ================== VISUALISASI ==================
elif menu == "Visualisasi Data":
    st.title("📈 Visualisasi Data")

    st.markdown("---")
    st.subheader("🗺️ Peta Sebaran AQI Berdasarkan Negara")
    fig1 = px.scatter_geo(data,
        lat="lat", lon="lng",
        color="AQI Category",
        hover_name="Country",
        projection="natural earth",
        title="Sebaran Kualitas Udara di Dunia"
    )
    st.plotly_chart(fig1, use_container_width=True)

    st.markdown("---")
    st.subheader("📊 Distribusi Nilai AQI dan Polutan")
    gas_columns = ['AQI Value', 'CO AQI Value', 'Ozone AQI Value', 'NO2 AQI Value', 'PM2.5 AQI Value']
    for col in gas_columns:
        fig, ax = plt.subplots()
        sns.histplot(data[col], kde=True, ax=ax, color="skyblue")
        ax.set_title(f"Distribusi {col}")
        st.pyplot(fig)

    st.markdown("---")
    st.subheader("🔗 Korelasi Antar Variabel Numerik")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(data.select_dtypes(include=np.number).corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    st.markdown("---")
    st.subheader("📦 Boxplot AQI berdasarkan Kategori")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=data, x='AQI Category', y='AQI Value', ax=ax, palette="pastel")
    st.pyplot(fig)

    st.markdown("---")
    st.subheader("📊 Distribusi Kategori AQI")
    fig, ax = plt.subplots()
    sns.countplot(data=data, x='AQI Category', order=data['AQI Category'].value_counts().index, ax=ax, palette="Set2")
    ax.set_title("Jumlah Negara per Kategori AQI")
    ax.tick_params(axis='x', rotation=45)
    st.pyplot(fig)

# ================== MODEL SVC ==================
elif menu == "Model SVC":
    st.title("🧠 Klasifikasi Kategori AQI dengan SVC")

    # Encoding label
    le = LabelEncoder()
    data['AQI_Label'] = le.fit_transform(data['AQI Category'])

    X = data[['AQI Value', 'CO AQI Value', 'Ozone AQI Value', 'NO2 AQI Value', 'PM2.5 AQI Value']]
    y = data['AQI_Label']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train model
    svc = SVC(kernel='rbf')
    svc.fit(X_train, y_train)
    y_pred = svc.predict(X_test)

    # Evaluasi
    acc = accuracy_score(y_test, y_pred)
    st.metric(label="🎯 Akurasi Model SVC", value=f"{acc * 100:.2f}%")

    st.subheader("📉 Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues",
                xticklabels=le.classes_, yticklabels=le.classes_)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    with st.expander("📄 Laporan Klasifikasi Lengkap"):
        st.text(classification_report(y_test, y_pred, target_names=le.classes_))
