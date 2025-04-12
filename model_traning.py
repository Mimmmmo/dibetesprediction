import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import pickle
import os

# Load dataset
df = pd.read_csv("data/diabetes.csv")

# Replace 0s with np.nan in specific columns
na_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[na_cols] = df[na_cols].replace(0, np.nan)

# Fill missing values with the column median
df.fillna(df.median(numeric_only=True), inplace=True)

# Feature and target
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train Linear Regression model
model = LinearRegression()
model.fit(X_scaled, y)

# Save model and scaler
os.makedirs("model", exist_ok=True)
pickle.dump(model, open("model/diabetes_model.pkl", "wb"))
pickle.dump(scaler, open("model/scaler.pkl", "wb"))

print("✅ Linear Regression model and scaler saved!")
