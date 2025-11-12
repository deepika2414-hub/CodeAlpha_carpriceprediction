# TASK 3: Car Price Prediction with Machine Learning (Using Kaggle Dataset)

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -------------------------------------
# Step 1: Load Dataset
# -------------------------------------
# Make sure 'Car details v3.csv' is in the same folder as this script
df = pd.read_csv('Car details v3.csv')
print("Initial Dataset Shape:", df.shape)
print(df.head(), "\n")

# -------------------------------------
# Step 2: Data Cleaning
# -------------------------------------
# Drop rows with missing values
df = df.dropna()

# Remove units and convert mileage to numeric
df['mileage'] = df['mileage'].str.replace(' kmpl', '').str.replace(' km/kg', '')
df['mileage'] = pd.to_numeric(df['mileage'], errors='coerce')

# Remove units and convert engine to numeric
df['engine'] = df['engine'].str.replace(' CC', '')
df['engine'] = pd.to_numeric(df['engine'], errors='coerce')

# Remove units and convert max_power to numeric
df['max_power'] = df['max_power'].str.replace(' bhp', '')
df['max_power'] = pd.to_numeric(df['max_power'], errors='coerce')

# Drop any remaining missing values
df = df.dropna()

# -------------------------------------
# Step 3: Feature Selection
# -------------------------------------
# Select useful columns
features = ['year', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner', 'mileage', 'engine', 'max_power']
target = 'selling_price'

X = df[features]
y = df[target]

# -------------------------------------
# Step 4: Encoding Categorical Features
# -------------------------------------
label_cols = ['fuel', 'seller_type', 'transmission', 'owner']
le = LabelEncoder()
for col in label_cols:
    X[col] = le.fit_transform(X[col])

# -------------------------------------
# Step 5: Feature Scaling
# -------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------------------
# Step 6: Train-Test Split
# -------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# -------------------------------------
# Step 7: Model Training
# -------------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# -------------------------------------
# Step 8: Model Evaluation
# -------------------------------------
y_pred = model.predict(X_test)

print("Actual Prices:", np.round(y_test.values[:5], 2))
print("Predicted Prices:", np.round(y_pred[:5], 2), "\n")

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error: {mae:.2f}")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R2 Score: {r2:.2f}\n")

# -------------------------------------
# Step 9: Visualization
# -------------------------------------
plt.figure(figsize=(8,5))
plt.scatter(y_test, y_pred, color='blue', edgecolors='black')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
plt.title("Actual vs Predicted Car Prices")
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.grid(True)
plt.show()
