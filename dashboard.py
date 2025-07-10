import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Analisis AQI Negara", layout="wide")

st.title("Analisis AQI dan Lokasi Negara Berdasarkan Dataset")

uploaded_file = st.file_uploader("Upload file CSV Anda", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    st.subheader("Data Awal")
    st.dataframe(data.head(10))

    st.subheader("Statistik Deskriptif")
    st.write(data.describe())

    st.subheader("Distribusi AQI (Air Quality Index)")
    fig, ax = plt.subplots()
    sns.histplot(data['AQI'], kde=True, ax=ax)
    st.pyplot(fig)

    st.subheader("Peta Sebaran Negara Berdasarkan Latitude dan Longitude")
    if 'Latitude' in data.columns and 'Longitude' in data.columns:
        st.map(data.rename(columns={"Latitude": "lat", "Longitude": "lon"}))
    else:
        st.warning("Kolom Latitude dan Longitude tidak ditemukan dalam data.")

else:
    st.info("Silakan upload file CSV untuk mulai analisis.")
