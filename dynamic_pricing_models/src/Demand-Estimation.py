# src/demand_estimation.py

import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# paths
processed_data_path = 'data/processed/sales_data.csv'
results_path = 'results/demand_estimation_results.txt'
figures_path = 'figures'

# create directories
os.makedirs(os.path.dirname(results_path), exist_ok=True)
os.makedirs(figures_path, exist_ok=True)

# load data
print("Loading processed data...")
data = pd.read_csv(processed_data_path)

# show few lines
print("Processed Data:")
print(data.head())

# rename columns to match expected names
data = data.rename(columns={
    'Price per Unit': 'price',
    'Quantity': 'sales'
})

# create rolling averages if needed (commented out since we don't have date info)
# data['date'] = pd.to_datetime(data['Date'])
# data = data.sort_values('date')
# data['rolling_avg_7'] = data['sales'].rolling(window=7).mean()
# data['rolling_std_7'] = data['sales'].rolling(window=7).std()
# data['sales_lag_1'] = data['sales'].shift(1)

# make sure columns are numbers
numeric_cols = ['price', 'sales']  # removed the rolling/lag columns for now
data[numeric_cols] = data[numeric_cols].apply(pd.to_numeric, errors='coerce')

# remove invalid rows
data = data[(data['price'] > 0) & (data['sales'] > 0)].copy()

# prepare data
X = data[['price']].values
y = data['sales'].values

# linear regression model
print("Fitting linear regression model...")
model = LinearRegression()
model.fit(X, y)

# Predict demand
y_pred = model.predict(X)

# Evaluate the model
mae = mean_absolute_error(y, y_pred)
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
r2 = model.score(X, y)

print("Model Performance:")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R-squared: {r2:.4f}")

with open(results_path, 'w') as f:
    f.write("Demand Estimation Model Performance:\n")
    f.write(f"MAE: {mae:.4f}\n")
    f.write(f"RMSE: {rmse:.4f}\n")
    f.write(f"R-squared: {r2:.4f}\n")

# Calculate demand elasticity
print("Calculating demand elasticity...")
data['price_log'] = np.log(data['price'])
data['sales_log'] = np.log(data['sales'])

# Remove missing values
data.replace([np.inf, -np.inf], np.nan, inplace=True)
data.dropna(subset=['price_log', 'sales_log'], inplace=True)

if len(data) > 1: #enough data to run regression
    try:
        elasticity_model = sm.OLS(data['sales_log'], sm.add_constant(data['price_log'])).fit()
        elasticity = elasticity_model.params['price_log']
        print(f"Demand Elasticity: {elasticity:.4f}")
        with open(results_path, 'a') as f:
            f.write(f"Demand Elasticity: {elasticity:.4f}\n")
    except Exception as e:
        print(f"Error calculating elasticity: {str(e)}")
        with open(results_path, 'a') as f:
            f.write("Could not calculate elasticity due to numerical errors\n")
else:
    print("Not enough valid observations to calculate elasticity")
    with open(results_path, 'a') as f:
        f.write("Could not calculate elasticity - insufficient valid data\n")

# relation between price and demand
plt.figure(figsize=(10, 6))
sns.scatterplot(x=data['price'], y=data['sales'])
plt.plot(data['price'], y_pred, color='red', linewidth=2)
plt.title('Price vs. Sales')
plt.xlabel('Price')
plt.ylabel('Sales')
plt.savefig(os.path.join(figures_path, 'price_vs_sales.png'))
plt.close()  

# log-log relationship 
if len(data) > 1 and 'price_log' in data.columns and 'sales_log' in data.columns:
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=data['price_log'], y=data['sales_log'])
    if len(data) > 1:
        try:
            plt.plot(data['price_log'], elasticity_model.fittedvalues, color='red', linewidth=2)
        except:
            pass
    plt.title('Log-Log Price vs. Sales')
    plt.xlabel('Log Price')
    plt.ylabel('Log Sales')
    plt.savefig(os.path.join(figures_path, 'log_price_vs_log_sales.png'))
    plt.close()
    
print("Demand estimation completed!")