import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

st.set_page_config(page_title="Analisis Kualitas Udara Dunia", layout="wide")
st.title("🌍 Analisis Kualitas Udara Berdasarkan Dataset Negara")

uploaded_file = st.file_uploader("📂 Upload file CSV Anda", type=["csv"])
if uploaded_file:
    data = pd.read_csv(uploaded_file)

    # Preprocessing
    st.subheader("📋 Data Awal")
    st.dataframe(data.head())

    st.subheader("📊 Statistik Deskriptif")
    st.write(data.describe(include="all"))

    # Cek dan hapus missing value
    st.subheader("🧼 Pemeriksaan Data Kosong")
    st.write(data.isnull().sum())
    data.dropna(inplace=True)
    st.success("✅ Data kosong berhasil dihapus")

    # Scatter plot lokasi
    st.subheader("🗺️ Sebaran Lokasi Negara")
    fig1 = px.scatter_geo(data,
        lat="lat", lon="lng",
        color="AQI Category",
        hover_name="Country",
        projection="natural earth",
        title="Sebaran Kualitas Udara Berdasarkan Negara"
    )
    st.plotly_chart(fig1, use_container_width=True)

    # Histogram distribusi AQI
    st.subheader("📈 Distribusi Nilai AQI dan Gas Terkait")
    gas_columns = ['AQI Value', 'CO AQI Value', 'Ozone AQI Value', 'NO2 AQI Value', 'PM2.5 AQI Value']
    for col in gas_columns:
        fig, ax = plt.subplots()
        sns.histplot(data[col], kde=True, ax=ax)
        ax.set_title(f"Distribusi {col}")
        st.pyplot(fig)

    # Korelasi
    st.subheader("🔗 Korelasi Antar Variabel Numerik")
    data_numeric = data.select_dtypes(include=[np.number])
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(data_numeric.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    # Boxplot
    st.subheader("📦 Boxplot AQI dan Gas Lainnya")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=data, x='AQI Category', y='AQI Value', ax=ax)
    ax.set_title("Persebaran AQI berdasarkan Kategori")
    st.pyplot(fig)

    # Kategori AQI
    st.subheader("📊 Distribusi Kategori AQI")
    fig, ax = plt.subplots()
    sns.countplot(data=data, x='AQI Category', order=data['AQI Category'].value_counts().index, ax=ax)
    ax.set_title("Jumlah Negara per Kategori AQI")
    ax.tick_params(axis='x', rotation=45)
    st.pyplot(fig)

    # Top kota dengan AQI tertinggi
    st.subheader("🏙️ 20 Kota dengan Rata-rata AQI Tertinggi")
    top_cities = data.groupby('City')['AQI Value'].mean().sort_values(ascending=False).head(20)
    st.dataframe(top_cities.reset_index())

else:
    st.info("Silakan upload file CSV untuk memulai analisis.")
