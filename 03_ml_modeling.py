"""
PHASE 3: MACHINE LEARNING MODELING
====================================

This script implements multiple regression models to predict average monthly
temperature using Year, Month, and NightLight intensity as predictors.

Academic Rationale:
- Multiple models allow comparison of different learning approaches
- Linear Regression: Baseline model, interpretable, assumes linear relationships
- Random Forest: Non-linear, handles interactions, robust to outliers
- Gradient Boosting: Sequential learning, often high performance, feature importance

Problem Definition:
- Type: Supervised Regression
- Inputs: Year, Month, NightLight (features/independent variables)
- Output: Avg_Monthly_Temperature (target/dependent variable)
- Goal: Learn the mapping f(Year, Month, NightLight) → Avg_Monthly_Temperature

Time-Based Split Rationale:
- Earlier months → Training (preserves temporal order)
- Recent months → Testing (simulates real-world prediction scenario)
- Prevents data leakage from future to past
- More realistic than random split for time series data

Step-by-step Process:
1. Load preprocessed monthly data
2. Prepare features (Year, Month, NightLight) and target
3. Time-based train-test split
4. Train multiple models
5. Save models for deployment
"""

import pandas as pd
import numpy as np
import pickle
import warnings
import os
warnings.filterwarnings('ignore')

# Create output directories if they don't exist
os.makedirs('outputs/csv', exist_ok=True)
os.makedirs('outputs/png', exist_ok=True)
os.makedirs('models', exist_ok=True)

# Machine Learning libraries
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ============================================================================
# STEP 1: LOAD PREPROCESSED DATA
# ============================================================================
print("="*70)
print("PHASE 3: MACHINE LEARNING MODELING")
print("="*70)

print("\nLoading preprocessed monthly dataset...")
df = pd.read_csv('outputs/csv/Chennai_Monthly_Aggregated.csv')
print(f"Dataset shape: {df.shape}")
print(f"\nDataset preview:")
print(df.head())

# ============================================================================
# STEP 2: PROBLEM DEFINITION
# ============================================================================
print("\n" + "="*70)
print("STEP 2: PROBLEM DEFINITION")
print("="*70)

"""
ACADEMIC PROBLEM STATEMENT:
---------------------------
Type: Supervised Regression Problem
- Supervised: We have labeled data (known temperature values)
- Regression: Target variable (temperature) is continuous

Input Features (Independent Variables):
- Year: Temporal variable (2014-2024)
- Month: Seasonal variable (1-12)
- NightLight: Human activity proxy (VIIRS intensity values)

Output Target (Dependent Variable):
- Avg_Monthly_Temperature: Average monthly temperature in °C

Objective:
Learn a function f such that: Avg_Monthly_Temperature = f(Year, Month, NightLight)

This function will allow us to:
1. Predict temperature for given year, month, and urbanization level
2. Understand the relative contribution of each feature
3. Quantify the urbanization-temperature relationship
"""

print("\nProblem Type: Supervised Regression")
print("Features (X): Year, Month, NightLight")
print("Target (y): Avg_Monthly_Temperature")
print(f"\nNumber of samples: {len(df)}")

# ============================================================================
# STEP 3: FEATURE ENGINEERING AND PREPARATION
# ============================================================================
print("\n" + "="*70)
print("STEP 3: FEATURE PREPARATION")
print("="*70)

"""
Academic Reasoning:
- Feature selection: We use Year, Month, and NightLight as predictors
- Year captures long-term trends (global warming, urbanization growth)
- Month captures seasonal patterns (Chennai's tropical climate has seasonal variations)
"""

# Extract features and target
X = df[['Year', 'Month', 'NightLight']].copy()  # Features (independent variables)
y = df['Avg_Monthly_Temperature'].copy()        # Target (dependent variable)

print("\nFeatures (X):")
print(X.head())
print(f"\nFeature statistics:")
print(X.describe())

print("\nTarget (y):")
print(y.head())
print(f"\nTarget statistics:")
print(y.describe())

# ============================================================================
# STEP 4: TIME-BASED TRAIN-TEST SPLIT
# ============================================================================
print("\n" + "="*70)
print("STEP 4: TIME-BASED TRAIN-TEST SPLIT")
print("="*70)

"""
Academic Reasoning for Time-Based Split:
-----------------------------------------
- Time-based split preserves temporal order (earlier → training, later → testing)
- Prevents data leakage: future data doesn't inform past predictions
- More realistic: simulates real-world scenario where we predict future based on past
- With monthly data (132 observations), we can use ~80% for training, ~20% for testing
- This gives us ~105 training samples and ~27 testing samples

Alternative to random split:
- Random split would mix past and future, violating temporal causality
- Time-based split respects the temporal nature of climate data
"""

# Sort by Year and Month to ensure chronological order
df_sorted = df.sort_values(['Year', 'Month']).reset_index(drop=True)
X_sorted = df_sorted[['Year', 'Month', 'NightLight']].copy()
y_sorted = df_sorted['Avg_Monthly_Temperature'].copy()

# Use 80% for training, 20% for testing (time-based)
split_index = int(len(df_sorted) * 0.8)

X_train = X_sorted.iloc[:split_index].copy()
X_test = X_sorted.iloc[split_index:].copy()
y_train = y_sorted.iloc[:split_index].copy()
y_test = y_sorted.iloc[split_index:].copy()

print(f"\nTime-based split:")
print(f"  Training set: {len(X_train)} samples ({len(X_train)/len(df_sorted)*100:.1f}%)")
print(f"  Testing set: {len(X_test)} samples ({len(X_test)/len(df_sorted)*100:.1f}%)")
print(f"\nTraining period: {X_train['Year'].min()}-{X_train['Month'].min():.0f} to {X_train['Year'].max()}-{X_train['Month'].max():.0f}")
print(f"Testing period: {X_test['Year'].min()}-{X_test['Month'].min():.0f} to {X_test['Year'].max()}-{X_test['Month'].max():.0f}")

# ============================================================================
# STEP 5: MODEL 1 - MULTIPLE LINEAR REGRESSION
# ============================================================================
print("\n" + "="*70)
print("STEP 5: MODEL 1 - LINEAR REGRESSION")
print("="*70)

"""
Academic Reasoning for Linear Regression:
-----------------------------------------
- Assumes linear relationship: y = β₀ + β₁*Year + β₂*Month + β₃*NightLight + ε
- Interpretable coefficients show direct feature contributions
- Baseline model for comparison
- Assumptions: linearity, independence, homoscedasticity, normality
"""

print("\nTraining Linear Regression model...")
print("-" * 50)

# Initialize and train Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Make predictions
y_train_pred_lr = lr_model.predict(X_train)
y_test_pred_lr = lr_model.predict(X_test)

# Display model coefficients (interpretability advantage)
print("\nModel Coefficients:")
print(f"  Intercept (β₀): {lr_model.intercept_:.4f}°C")
print(f"  Year coefficient (β₁): {lr_model.coef_[0]:.4f}°C/year")
print(f"  Month coefficient (β₂): {lr_model.coef_[1]:.4f}°C/month")
print(f"  NightLight coefficient (β₃): {lr_model.coef_[2]:.4f}°C/unit")

print("\nInterpretation:")
print(f"  - Each additional year increases temperature by {lr_model.coef_[0]:.4f}°C")
print(f"  - Each additional month (seasonal) changes temperature by {lr_model.coef_[1]:.4f}°C")
print(f"  - Each unit increase in NightLight increases temperature by {lr_model.coef_[2]:.4f}°C")

# Save model
with open('models/model_linear_regression.pkl', 'wb') as f:
    pickle.dump(lr_model, f)
print("\n✓ Model saved: models/model_linear_regression.pkl")

# ============================================================================
# STEP 6: MODEL 2 - RANDOM FOREST REGRESSOR
# ============================================================================
print("\n" + "="*70)
print("STEP 6: MODEL 2 - RANDOM FOREST REGRESSOR")
print("="*70)

"""
Academic Reasoning for Random Forest:
-------------------------------------
- Ensemble method: combines multiple decision trees
- Handles non-linear relationships and feature interactions
- Robust to outliers and overfitting (via averaging and regularization)
- Provides feature importance scores
- No assumptions about data distribution
- Regularization parameters help control overfitting
"""

print("\nTraining Random Forest Regressor...")
print("-" * 50)

# Initialize Random Forest with hyperparameters (regularized to prevent overfitting)
rf_model = RandomForestRegressor(
    n_estimators=100,      # Number of trees in forest
    max_depth=10,           # Limit tree depth to prevent overfitting
    min_samples_split=5,    # Minimum samples to split a node (regularization)
    min_samples_leaf=2,     # Minimum samples in a leaf (regularization)
    random_state=42,        # Reproducibility
    n_jobs=-1               # Use all available cores
)

rf_model.fit(X_train, y_train)

# Make predictions
y_train_pred_rf = rf_model.predict(X_train)
y_test_pred_rf = rf_model.predict(X_test)

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance)

# Save model
with open('models/model_random_forest.pkl', 'wb') as f:
    pickle.dump(rf_model, f)
print("\n✓ Model saved: models/model_random_forest.pkl")

# ============================================================================
# STEP 7: MODEL 3 - GRADIENT BOOSTING REGRESSOR
# ============================================================================
print("\n" + "="*70)
print("STEP 7: MODEL 3 - GRADIENT BOOSTING REGRESSOR")
print("="*70)

"""
Academic Reasoning for Gradient Boosting:
-----------------------------------------
- Sequential ensemble: each tree corrects errors of previous trees
- Often achieves high predictive performance
- Handles complex non-linear patterns
- Provides feature importance
- Regularization parameters help control overfitting
"""

print("\nTraining Gradient Boosting Regressor...")
print("-" * 50)

# Initialize Gradient Boosting with hyperparameters (regularized)
gb_model = GradientBoostingRegressor(
    n_estimators=100,      # Number of boosting stages
    learning_rate=0.1,     # Shrinkage factor (lower = more conservative, prevents overfitting)
    max_depth=5,           # Maximum depth of trees (regularization)
    min_samples_split=5,   # Minimum samples to split (regularization)
    min_samples_leaf=2,    # Minimum samples in leaf (regularization)
    random_state=42
)

gb_model.fit(X_train, y_train)

# Make predictions
y_train_pred_gb = gb_model.predict(X_train)
y_test_pred_gb = gb_model.predict(X_test)

# Feature importance
feature_importance_gb = pd.DataFrame({
    'Feature': X.columns,
    'Importance': gb_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance_gb)

# Save model
with open('models/model_gradient_boosting.pkl', 'wb') as f:
    pickle.dump(gb_model, f)
print("\n✓ Model saved: models/model_gradient_boosting.pkl")

# ============================================================================
# STEP 8: WHY MULTIPLE MODELS?
# ============================================================================
print("\n" + "="*70)
print("STEP 8: RATIONALE FOR MULTIPLE MODELS")
print("="*70)

"""
ACADEMIC JUSTIFICATION FOR MODEL COMPARISON:
--------------------------------------------

1. Model Diversity:
   - Linear: Assumes linearity, interpretable
   - Random Forest: Non-linear, robust ensemble
   - Gradient Boosting: Sequential learning, high performance potential

2. Bias-Variance Trade-off:
   - Different models balance bias and variance differently
   - Comparing them helps select optimal trade-off

3. Robustness Testing:
   - Consistent results across models strengthen findings
   - Disagreement indicates model-specific assumptions

4. Best Practice:
   - Standard ML workflow: compare multiple models
   - Ensures we're not missing better approaches

5. Feature Importance Comparison:
   - Different models may highlight different important features
   - Consensus across models provides stronger evidence

6. Overfitting Control:
   - Multiple models with regularization help identify overfitting
   - Compare training vs testing performance across models
"""

print("\nModel comparison rationale:")
print("✓ Linear Regression: Baseline, interpretable")
print("✓ Random Forest: Non-linear, robust, regularized")
print("✓ Gradient Boosting: Sequential learning, high performance, regularized")

# ============================================================================
# STEP 9: SAVE DATA SPLITS FOR EVALUATION
# ============================================================================
print("\n" + "="*70)
print("STEP 9: SAVING DATA SPLITS")
print("="*70)

# Save train-test splits for evaluation script
X_train.to_csv('outputs/csv/X_train.csv', index=False)
X_test.to_csv('outputs/csv/X_test.csv', index=False)
y_train.to_csv('outputs/csv/y_train.csv', index=False)
y_test.to_csv('outputs/csv/y_test.csv', index=False)

print("✓ Training and testing splits saved")
print(f"  - outputs/csv/X_train.csv, outputs/csv/X_test.csv")
print(f"  - outputs/csv/y_train.csv, outputs/csv/y_test.csv")

print("\n" + "="*70)
print("MACHINE LEARNING MODELING COMPLETE")
print("="*70)
print("\nNext step: Model evaluation and comparison (Phase 4)")
