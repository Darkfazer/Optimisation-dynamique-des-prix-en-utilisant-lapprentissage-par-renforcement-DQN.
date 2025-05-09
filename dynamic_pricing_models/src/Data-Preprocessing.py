# src/data_preprocessing.py

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from datetime import datetime

# paths
raw_data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'sales_data.csv')
processed_data_path = 'data/processed/processed_data.csv'

# craete a folder if they doesnt exist
os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)

# Read data
print("Loading raw data...")
data = pd.read_csv(raw_data_path)

# show firsts line
print("Raw Data:")
print(data.head())


print("Cleaning data...")

# Handle missing values
data.fillna(method='ffill', inplace=True)

# Rename columns 
data = data.rename(columns={
    'Date': 'date',
    'Quantity': 'sales',
    'Price per Unit': 'price'
})

# date column to datetime
data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d')


print("Performing feature engineering...")

# temporal features
data['year'] = data['date'].dt.year
data['month'] = data['date'].dt.month
data['day'] = data['date'].dt.day
data['day_of_week'] = data['date'].dt.dayofweek
data['is_weekend'] = data['day_of_week'].isin([5, 6]).astype(int)

# Price features
data['price_diff'] = data['price'].diff().fillna(0)
data['price_change_pct'] = data['price'].pct_change().fillna(0)

# Sales features
data['rolling_avg_7'] = data['sales'].rolling(window=7).mean().fillna(0)
data['rolling_std_7'] = data['sales'].rolling(window=7).std().fillna(0)
data['sales_lag_1'] = data['sales'].shift(1).fillna(0)

# Customer grpd by age
data['customer_segment'] = pd.qcut(data['Age'], q=4, labels=False)

# Encode categorical features
print("Encoding categorical features...")
label_encoders = {}
categorical_cols = ['Gender', 'Product Category', 'Customer ID']
for column in categorical_cols:
    if column in data.columns:
        label_encoders[column] = LabelEncoder() # Creates an encoder
        data[column] = label_encoders[column].fit_transform(data[column]) # T to N

# Normalize numerical 0/1
print("Normalizing numerical features...")
scaler = MinMaxScaler()
numerical_features = [
    'sales', 
    'price', 
    'price_diff',
    'price_change_pct',
    'rolling_mean_7',
    'rolling_std_7',
    'sales_lag_1',
    'Age'
]
data[numerical_features] = scaler.fit_transform(data[numerical_features])

# Save processed data
print("Saving processed data...")
data.to_csv(processed_data_path, index=False)

# Save encoders and scaler for inference
import joblib
os.makedirs('models/preprocessing', exist_ok=True)
joblib.dump(label_encoders, 'models/preprocessing/label_encoders.pkl')
joblib.dump(scaler, 'models/preprocessing/scaler.pkl')

print("Data preprocessing completed!")
print(f"Processed data saved to: {processed_data_path}")