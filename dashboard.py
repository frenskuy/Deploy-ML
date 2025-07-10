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

# Setup halaman
st.set_page_config(page_title="Dashboard AQI Dunia", layout="wide")
st.title("🌍 Dashboard Analisis Kualitas Udara Dunia")

# Navigasi di bagian atas
menu = st.sidebar.radio("📁 Pilih Halaman:", ["Beranda", "Eksplorasi Data", "Visualisasi", "Model Klasifikasi"])

# Load dan bersihkan data
DATA_PATH = "AQI and Lat Long of Countries.csv"
data = pd.read_csv(DATA_PATH)
data.dropna(inplace=True)

# ===================== BERANDA =====================
if menu == "Beranda":
    st.markdown("""
    ### 👋 Selamat datang!
    Dashboard ini menyediakan eksplorasi dan analisis data kualitas udara dari berbagai kota di dunia berdasarkan **Air Quality Index (AQI)**.

    **Fitur:**
    - Eksplorasi dan statistik deskriptif data
    - Visualisasi distribusi AQI dan polutan
    - Klasifikasi kategori AQI menggunakan model **Support Vector Classifier (SVC)**

    ---
    """)

# ===================== EKSPLORASI =====================
elif menu == "Eksplorasi Data":
    st.markdown("### 🔍 Eksplorasi Data")

    st.markdown("#### 📋 Tabel Data")
    st.dataframe(data.head(), use_container_width=True)

    st.markdown("#### 📊 Statistik Deskriptif")
    st.write(data.describe(include="all"))

    st.markdown("#### 🧼 Jumlah Data Kosong")
    st.write(data.isnull().sum())

    st.markdown("#### 🏙️ 20 Kota dengan AQI Rata-rata Tertinggi")
    top_cities = data.groupby('City')['AQI Value'].mean().sort_values(ascending=False).head(20)
    st.dataframe(top_cities.reset_index(), use_container_width=True)

# ===================== VISUALISASI =====================
elif menu == "Visualisasi":
    st.markdown("### 📈 Visualisasi Data")

    st.markdown("#### 🗺️ Peta Sebaran Kualitas Udara")
    fig_map = px.scatter_geo(data,
        lat="lat", lon="lng",
        color="AQI Category",
        hover_name="Country",
        projection="natural earth",
        title="Sebaran Negara Berdasarkan Kategori AQI"
    )
    st.plotly_chart(fig_map, use_container_width=True)

    st.markdown("#### 📉 Distribusi Nilai AQI dan Polutan")
    for col in ['AQI Value', 'CO AQI Value', 'Ozone AQI Value', 'NO2 AQI Value', 'PM2.5 AQI Value']:
        fig, ax = plt.subplots()
        sns.histplot(data[col], kde=True, ax=ax, color="skyblue")
        ax.set_title(f"Distribusi {col}", fontsize=16)
        ax.set_xlabel(col, fontsize=12)
        ax.set_ylabel("Frekuensi", fontsize=12)
        st.pyplot(fig)

    st.markdown("#### 🔗 Korelasi Antar Variabel")
    fig_corr, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(data.select_dtypes(include=np.number).corr(), annot=True, cmap="coolwarm", ax=ax)
    ax.set_title("Korelasi Antar Variabel", fontsize=16)
    st.pyplot(fig_corr)

    st.markdown("#### 📦 Boxplot AQI berdasarkan Kategori")
    fig_box, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=data, x='AQI Category', y='AQI Value', ax=ax, palette="pastel")
    ax.set_title("Boxplot AQI berdasarkan Kategori", fontsize=16)
    st.pyplot(fig_box)

    st.markdown("#### 📊 Jumlah Negara per Kategori AQI")
    fig, ax = plt.subplots()
    sns.countplot(data=data, x='AQI Category', order=data['AQI Category'].value_counts().index, ax=ax, palette="Set2")
    ax.set_title("Distribusi Kategori AQI", fontsize=16)
    ax.tick_params(axis='x', rotation=45)
    st.pyplot(fig)

# ===================== MODEL SVC =====================
elif menu == "Model Klasifikasi":
    st.markdown("### 🤖 Model Klasifikasi Kategori AQI dengan SVC")

    le = LabelEncoder()
    data['AQI_Label'] = le.fit_transform(data['AQI Category'])

    X = data[['AQI Value', 'CO AQI Value', 'Ozone AQI Value', 'NO2 AQI Value', 'PM2.5 AQI Value']]
    y = data['AQI_Label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    svc = SVC(kernel='rbf')
    svc.fit(X_train, y_train)
    y_pred = svc.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    st.metric("🎯 Akurasi Model", f"{acc * 100:.2f}%")

    st.markdown("#### 📉 Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues",
                xticklabels=le.classes_, yticklabels=le.classes_)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=16)
    st.pyplot(fig)

    with st.expander("📄 Laporan Klasifikasi Lengkap"):
        st.text(classification_report(y_test, y_pred, target_names=le.classes_))
