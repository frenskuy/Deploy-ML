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

# Setup halaman dengan tampilan wide
st.set_page_config(page_title="🌍 Dashboard AQI Dunia", layout="wide", page_icon="🌍")

# CSS untuk styling tabs
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0 25px;
        background-color: #F0F2F6;
        border-radius: 10px 10px 0px 0px;
        font-weight: bold;
    }

    .stTabs [aria-selected="true"] {
        background-color: #4CAF50;
        color: white;
    }

    .stTabs [data-baseweb="tab-highlight"] {
        background-color: transparent;
    }
    
    div[data-testid="stMetricValue"] {
        font-size: 24px;
    }
</style>
""", unsafe_allow_html=True)

# Header dengan judul dan deskripsi
st.title("🌍 Dashboard Analisis Kualitas Udara Dunia")
st.markdown("""
<div style="margin-bottom: 20px; padding: 10px; background-color: #f8f9fa; border-radius: 10px;">
    Visualisasi interaktif kualitas udara global berdasarkan Air Quality Index (AQI) dan berbagai parameter polutan.
</div>
""", unsafe_allow_html=True)

# Membuat tabs di bagian atas
tab1, tab2, tab3, tab4 = st.tabs(["🏠 Beranda", "🔍 Eksplorasi Data", "📊 Visualisasi", "🤖 Model Prediksi"])

# Load dan bersihkan data
DATA_PATH = "AQI and Lat Long of Countries.csv"
data = pd.read_csv(DATA_PATH)
data.dropna(inplace=True)

# ===================== TAB BERANDA =====================
with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 👋 Selamat Datang di Dashboard AQI Global!
        """)
        st.image("https://placehold.co/600x300", caption="Peta Kualitas Udara Global")
        
    with col2:
        st.markdown("""
        #### 📊 Tentang Dashboard Ini:
        
        - Analisis data AQI dari berbagai negara
        - Visualisasi interaktif menggunakan Plotly
        - Prediksi kategori AQI dengan Machine Learning
        
        """)
        
        st.info("""
        ℹ️ **Kategori AQI:**
        - Baik: 0-50
        - Sedang: 51-100
        - Tidak sehat: 101-150
        - Sangat tidak sehat: 151-200
        - Berbahaya: >200
        """)

# ===================== TAB EKSPLORASI ==================== 
with tab2:
    st.subheader("🔍 Eksplorasi Data")
    
    with st.expander("📋 Tampilkan Data", expanded=True):
        st.dataframe(data.head(), use_container_width=True)
    
    cols = st.columns(2)
    with cols[0]:
        st.metric("Jumlah Data", len(data))
        st.write("Statistik Deskriptif AQI:", data['AQI Value'].describe())
    
    with cols[1]:
        st.metric("Jumlah Negara", data['Country'].nunique())
        st.write("Top 5 Kota:", data['City'].value_counts().head())

# ===================== TAB VISUALISASI ====================
with tab3:
    st.subheader("📊 Visualisasi Interaktif")
    
    plot_type = st.radio("Pilih Jenis Visualisasi:", 
                        ["Peta Sebaran", "Distribusi AQI", "Korelasi Polutan"],
                        horizontal=True)
    
    if plot_type == "Peta Sebaran":
        fig = px.scatter_geo(data, 
                           lat="lat", lon="lng",
                           color="AQI Category",
                           hover_name="Country",
                           size="AQI Value",
                           projection="natural earth")
        st.plotly_chart(fig, use_container_width=True)
    
    elif plot_type == "Distribusi AQI":
        fig, ax = plt.subplots()
        sns.histplot(data['AQI Value'], kde=True, color='green')
        st.pyplot(fig)
    
    else:
        fig, ax = plt.subplots()
        sns.heatmap(data[['AQI Value','CO AQI Value','Ozone AQI Value']].corr(), 
                   annot=True, cmap="coolwarm")
        st.pyplot(fig)

# ===================== TAB MODEL ==================== 
with tab4:
    st.subheader("🤖 Model Prediksi Kategori AQI")
    
    # Prepare data
    le = LabelEncoder()
    data['AQI_Label'] = le.fit_transform(data['AQI Category'])
    
    X = data[['AQI Value', 'CO AQI Value', 'Ozone AQI Value']]
    y = data['AQI_Label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train model
    model = SVC(kernel='rbf')
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    
    # Display results
    cols = st.columns(3)
    cols[0].metric("Akurasi Model", f"{acc*100:.1f}%")
    cols[1].metric("Jumlah Training", len(X_train))
    cols[2].metric("Jumlah Testing", len(X_test))
    
    # Confusion matrix
    st.subheader("Matriks Kebingungan")
    y_pred = model.predict(X_test)
    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, y_pred), 
               annot=True, fmt="d", 
               xticklabels=le.classes_, 
               yticklabels=le.classes_)
    st.pyplot(fig)
