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

st.set_page_config(page_title="Analisis Kualitas Udara Dunia", layout="wide")
st.title("🌍 Analisis Kualitas Udara Berdasarkan Dataset Negara")

# Load file CSV langsung
DATA_PATH = "AQI and Lat Long of Countries.csv"
data = pd.read_csv(DATA_PATH)

# --- Pra-pemrosesan ---
data.dropna(inplace=True)

# --- Tampilan Awal ---
st.subheader("📋 Data Awal")
st.dataframe(data.head())

# --- Visualisasi Sebaran AQI ---
st.subheader("🗺️ Sebaran Lokasi Negara")
fig1 = px.scatter_geo(data,
    lat="lat", lon="lng",
    color="AQI Category",
    hover_name="Country",
    projection="natural earth",
    title="Sebaran Kualitas Udara Berdasarkan Negara"
)
st.plotly_chart(fig1, use_container_width=True)

# --- Distribusi Variabel ---
st.subheader("📈 Distribusi Nilai AQI dan Gas Terkait")
for col in ['AQI Value', 'CO AQI Value', 'Ozone AQI Value', 'NO2 AQI Value', 'PM2.5 AQI Value']:
    fig, ax = plt.subplots()
    sns.histplot(data[col], kde=True, ax=ax)
    ax.set_title(f"Distribusi {col}")
    st.pyplot(fig)

# --- Korelasi ---
st.subheader("🔗 Korelasi Antar Variabel")
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(data.select_dtypes(include=np.number).corr(), annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)

# --- Boxplot ---
st.subheader("📦 Boxplot AQI vs Kategori")
fig, ax = plt.subplots(figsize=(12, 6))
sns.boxplot(data=data, x='AQI Category', y='AQI Value', ax=ax)
st.pyplot(fig)

# --- Distribusi Kategori ---
st.subheader("📊 Distribusi Kategori AQI")
fig, ax = plt.subplots()
sns.countplot(data=data, x='AQI Category', order=data['AQI Category'].value_counts().index, ax=ax)
ax.tick_params(axis='x', rotation=45)
st.pyplot(fig)

# --- Top Kota Tertinggi AQI ---
st.subheader("🏙️ 20 Kota dengan AQI Tertinggi")
top_cities = data.groupby('City')['AQI Value'].mean().sort_values(ascending=False).head(20)
st.dataframe(top_cities.reset_index())

# ===============================
# === MODELLING SVC (Klasifikasi AQI Category) ===
# ===============================
st.subheader("🧠 Klasifikasi Kategori AQI dengan SVC")

# Encode target label
le = LabelEncoder()
data['AQI_Label'] = le.fit_transform(data['AQI Category'])

# Fitur dan target
X = data[['AQI Value', 'CO AQI Value', 'Ozone AQI Value', 'NO2 AQI Value', 'PM2.5 AQI Value']]
y = data['AQI_Label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
svc = SVC(kernel='rbf')
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)

# Hasil evaluasi
acc = accuracy_score(y_test, y_pred)
st.metric(label="🎯 Akurasi Model SVC", value=f"{acc * 100:.2f}%")

# Optional: tampilkan laporan klasifikasi
with st.expander("📄 Laporan Klasifikasi Lengkap"):
    st.text(classification_report(y_test, y_pred, target_names=le.classes_))

# Confusion Matrix
st.subheader("📉 Confusion Matrix")
fig, ax = plt.subplots()
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues",
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel("Predicted")
plt.ylabel("True")
st.pyplot(fig)
