import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import joblib

# Load data
data = pd.read_csv("AQI and Lat Long of Countries.csv")
data.dropna(inplace=True)

# Encode target
le = LabelEncoder()
data['AQI_Label'] = le.fit_transform(data['AQI Category'])

# Fitur dan target
X = data[['AQI Value', 'CO AQI Value', 'Ozone AQI Value', 'NO2 AQI Value', 'PM2.5 AQI Value']]
y = data['AQI_Label']

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
svc = SVC(kernel='rbf')
svc.fit(X_train, y_train)

# Save model dan encoder
joblib.dump(svc, "model_svc.pkl")
joblib.dump(le, "label_encoder.pkl")
print("✅ Model dan encoder berhasil disimpan.")
