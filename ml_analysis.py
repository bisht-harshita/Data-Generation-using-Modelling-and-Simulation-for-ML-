"""
Machine Learning Analysis of Simulated Data
===========================================
This script takes the generated 'simulation_data.csv' and applies various ML models to predict 'Average_Wait_Time'.

Methodology:
1. Load dataset (Feature Engineering if needed).
2. Split into Train/Test sets (80/20).
3. Train multiple models (Linear, Tree-based, Ensemble, SVR, etc.).
4. Evaluate using RMSE (Root Mean Squared Error) and R2 Score.
5. Create a comparison table and "Actual vs Predicted" plot for the best model.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
from tabulate import tabulate

# Load Data
df = pd.read_csv('simulation_data.csv')

# Features (X) and Target (y)
X = df[['Num_Agents', 'Arrival_Interval', 'Service_Time']]
y = df['Average_Wait_Time']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling (Important for SVR, KNN, Linear Models)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Models Dictionary
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(),
    'Lasso Regression': Lasso(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42),
    'XGBoost': xgb.XGBRegressor(objective='reg:squarederror', random_state=42),
    'AdaBoost': AdaBoostRegressor(random_state=42),
    'SVR': SVR(),
    'KNN': KNeighborsRegressor()
}

results = []

print("Training Models...")

best_model = None
best_score = -np.inf
best_model_name = ""

for name, model in models.items():
    # Use scaled data for non-tree models (heuristically applied to all for simplicity or selectively)
    # Tree models don't need scaling, but it doesn't hurt. SVR/KNN/Linear do.
    # We will use scaled data for all for consistency in this script.
    
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    results.append({
        'Model': name,
        'RMSE': rmse,
        'R2 Score': r2
    })
    
    if r2 > best_score:
        best_score = r2
        best_model = model
        best_model_name = name

# Create Comparison Table
results_df = pd.DataFrame(results).sort_values(by='R2 Score', ascending=False)

print("\nModel Comparison:")
print(tabulate(results_df, headers='keys', tablefmt='grid', floatfmt=".4f"))

# Visualization: Actual vs Predicted for Best Model
y_pred_best = best_model.predict(X_test_scaled)

plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred_best, alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.xlabel('Actual Wait Time')
plt.ylabel(f'Predicted Wait Time ({best_model_name})')
plt.title(f'Actual vs Predicted - {best_model_name} (R2: {best_score:.4f})')
plt.tight_layout()
plt.savefig('actual_vs_predicted.png')
print(f"\nSaved prediction plot to 'actual_vs_predicted.png'")

# Feature Importance (if applicable)
if hasattr(best_model, 'feature_importances_'):
    plt.figure(figsize=(8, 5))
    importances = best_model.feature_importances_
    features = X.columns
    sns.barplot(x=importances, y=features)
    plt.title(f'Feature Importance - {best_model_name}')
    plt.savefig('feature_importance.png')
    print("Saved feature importance plot to 'feature_importance.png'")

# Save results to CSV
results_df.to_csv('model_results.csv', index=False)
print("Saved model results to 'model_results.csv'")
