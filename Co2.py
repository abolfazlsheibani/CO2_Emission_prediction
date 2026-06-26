# CO2 Emission Prediction
# Predicts vehicle CO2 emissions using engine size, cylinders, and fuel type
# Dataset: 500 vehicles | Model: Linear Regression | R² = 0.80 | MAE = 23.3

import numpy as np
import pandas as pd
from sklearn import linear_model
import seaborn as sns
import sklearn.metrics as sm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('co2.csv')

# Data overview
print("=" * 50)
print("Dataset Info:")
print(df.info())
print("\nDataset Description:")
print(df.describe())

# Visualize target variable distribution
plt.figure(figsize=(8, 5))
sns.histplot(df['out1'], bins=30, kde=True, color='steelblue')
plt.title('Distribution of CO2 Emissions')
plt.xlabel('CO2 Emission')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('co2_distribution.png')
plt.show()

# Correlation heatmap
plt.figure(figsize=(9, 9))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.savefig('correlation_heatmap.png')
plt.show()

# Define features and target
x = df[['engine', 'cylandr', 'fuelcomb']]
y = df['out1']

print("\nFeatures used:", list(x.columns))
print("Target: out1 (CO2 Emissions)")

# Train/test split (80% train, 20% test)
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=42)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# Train Linear Regression model
model = linear_model.LinearRegression()
model.fit(X_train, Y_train)

# Model coefficients
print("\n" + "=" * 50)
print("Model Coefficients:")
for feature, coef in zip(x.columns, model.coef_):
    print(f"  {feature}: {coef:.4f}")
print(f"  Intercept: {model.intercept_:.4f}")

# Make predictions
y_pred = model.predict(X_test)

# Compare actual vs predicted (first 10 samples)
comparison = pd.DataFrame({'Actual': Y_test.values, 'Predicted': y_pred})
print("\n" + "=" * 50)
print("Actual vs Predicted (first 10 samples):")
print(comparison.head(10).to_string(index=False))

# Model evaluation
r2 = sm.r2_score(Y_test, y_pred)
mae = sm.mean_absolute_error(Y_test, y_pred)
mse = sm.mean_squared_error(Y_test, y_pred)
rmse = np.sqrt(mse)

print("\n" + "=" * 50)
print("Model Performance:")
print(f"  R² Score:  {r2:.4f}")
print(f"  MAE:       {mae:.2f}")
print(f"  MSE:       {mse:.2f}")
print(f"  RMSE:      {rmse:.2f}")

# Actual vs Predicted scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(Y_test, y_pred, alpha=0.6, color='steelblue', edgecolors='white', s=60)
plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], '--r', linewidth=2, label='Perfect Prediction')
plt.xlabel('Actual CO2 Emission', fontsize=12)
plt.ylabel('Predicted CO2 Emission', fontsize=12)
plt.title(f'Actual vs Predicted CO2 Emissions (R² = {r2:.3f})', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('actual_vs_predicted.png')
plt.show()

# Residual plot
residuals = Y_test - y_pred
plt.figure(figsize=(8, 5))
plt.scatter(y_pred, residuals, alpha=0.6, color='teal', edgecolors='white', s=60)
plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
plt.xlabel('Predicted CO2 Emission', fontsize=12)
plt.ylabel('Residual (Actual - Predicted)', fontsize=12)
plt.title('Residual Plot', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('residual_plot.png')
plt.show()

print("\n" + "=" * 50)
print("All plots saved as PNG files.")
print("Model training complete.")
