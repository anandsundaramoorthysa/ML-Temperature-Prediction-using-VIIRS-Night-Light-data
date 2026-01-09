"""
PHASE 4: MODEL EVALUATION & COMPARISON
=======================================

This script evaluates all trained models using multiple metrics and creates
a comprehensive comparison table to identify the best-performing model.

Academic Rationale:
- Model evaluation is critical for selecting the best model
- Multiple metrics provide different perspectives on model performance
- RMSE: Penalizes large errors (sensitive to outliers)
- MAE: Average error magnitude (robust to outliers)
- R²: Proportion of variance explained (goodness of fit)
- Overfitting Control: Compare training vs testing performance to detect overfitting
- Model Selection: Balance between accuracy, interpretability, and generalization

Evaluation Metrics Explained:
1. RMSE (Root Mean Squared Error): sqrt(mean((y_true - y_pred)²))
   - Same units as target variable
   - Larger errors have disproportionately more impact
   
2. MAE (Mean Absolute Error): mean(|y_true - y_pred|)
   - Easier to interpret (average error)
   - All errors weighted equally
   
3. R² Score (Coefficient of Determination): 1 - (SS_res / SS_tot)
   - Ranges from -∞ to 1 (1 = perfect, 0 = baseline, <0 = worse than mean)
   - Proportion of variance in target explained by model

Step-by-step Process:
1. Load models and test data
2. Generate predictions for all models (both training and testing)
3. Calculate evaluation metrics (RMSE, MAE, R²)
4. Compare training vs testing performance (overfitting detection)
5. Create comparison table
6. Visualize model performance
7. Select best model based on metrics and overfitting control
"""

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Create output directories if they don't exist
os.makedirs('outputs/csv', exist_ok=True)
os.makedirs('outputs/png', exist_ok=True)

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# ============================================================================
# STEP 1: LOAD MODELS AND TEST DATA
# ============================================================================
print("="*70)
print("PHASE 4: MODEL EVALUATION & COMPARISON")
print("="*70)

print("\nLoading models and test data...")
print("-" * 50)

# Load models
with open('models/model_linear_regression.pkl', 'rb') as f:
    lr_model = pickle.load(f)

with open('models/model_random_forest.pkl', 'rb') as f:
    rf_model = pickle.load(f)

with open('models/model_gradient_boosting.pkl', 'rb') as f:
    gb_model = pickle.load(f)

print("✓ All models loaded successfully")

# Load test data
X_test = pd.read_csv('outputs/csv/X_test.csv')
y_test = pd.read_csv('outputs/csv/y_test.csv')
y_test = y_test.iloc[:, 0]  # Extract series from DataFrame

# Load training data for overfitting detection
X_train = pd.read_csv('outputs/csv/X_train.csv')
y_train = pd.read_csv('outputs/csv/y_train.csv')
y_train = y_train.iloc[:, 0]  # Extract series from DataFrame

print(f"✓ Test data loaded: {len(X_test)} samples")
print(f"✓ Training data loaded: {len(X_train)} samples (for overfitting detection)")

# ============================================================================
# STEP 2: GENERATE PREDICTIONS
# ============================================================================
print("\n" + "="*70)
print("STEP 2: GENERATING PREDICTIONS")
print("="*70)

print("\nGenerating predictions for all models...")
print("-" * 50)

# Generate predictions for each model (both training and testing)
y_train_pred_lr = lr_model.predict(X_train)
y_test_pred_lr = lr_model.predict(X_test)

y_train_pred_rf = rf_model.predict(X_train)
y_test_pred_rf = rf_model.predict(X_test)

y_train_pred_gb = gb_model.predict(X_train)
y_test_pred_gb = gb_model.predict(X_test)

print("✓ Predictions generated for:")
print("  - Linear Regression (training & testing)")
print("  - Random Forest (training & testing)")
print("  - Gradient Boosting (training & testing)")

# ============================================================================
# STEP 3: CALCULATE EVALUATION METRICS
# ============================================================================
print("\n" + "="*70)
print("STEP 3: CALCULATING EVALUATION METRICS")
print("="*70)

"""
Academic Reasoning for Metric Selection:
-----------------------------------------
1. RMSE: Emphasizes larger errors, useful when large deviations are critical
2. MAE: Provides interpretable average error, less sensitive to outliers
3. R²: Measures model's explanatory power relative to baseline (mean)

Using multiple metrics provides a comprehensive view of model performance.
"""

def calculate_metrics(y_true, y_pred, model_name):
    """
    Calculate RMSE, MAE, and R² for a model.
    
    Parameters:
    -----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted target values
    model_name : str
        Name of the model (for display)
    
    Returns:
    --------
    dict : Dictionary containing all metrics
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return {
        'Model': model_name,
        'RMSE': rmse,
        'MAE': mae,
        'R² Score': r2
    }

# Calculate metrics for each model
print("\nCalculating metrics...")
print("-" * 50)

# Calculate metrics for testing set
metrics_lr = calculate_metrics(y_test, y_test_pred_lr, 'Linear Regression')
metrics_rf = calculate_metrics(y_test, y_test_pred_rf, 'Random Forest')
metrics_gb = calculate_metrics(y_test, y_test_pred_gb, 'Gradient Boosting')

# Calculate metrics for training set (for overfitting detection)
metrics_lr_train = calculate_metrics(y_train, y_train_pred_lr, 'Linear Regression (Train)')
metrics_rf_train = calculate_metrics(y_train, y_train_pred_rf, 'Random Forest (Train)')
metrics_gb_train = calculate_metrics(y_train, y_train_pred_gb, 'Gradient Boosting (Train)')

# Display individual model results
print("\nLinear Regression Metrics:")
print(f"  RMSE: {metrics_lr['RMSE']:.4f}°C")
print(f"  MAE:  {metrics_lr['MAE']:.4f}°C")
print(f"  R²:   {metrics_lr['R² Score']:.4f}")

print("\nRandom Forest Metrics:")
print(f"  RMSE: {metrics_rf['RMSE']:.4f}°C")
print(f"  MAE:  {metrics_rf['MAE']:.4f}°C")
print(f"  R²:   {metrics_rf['R² Score']:.4f}")

print("\nGradient Boosting Metrics:")
print(f"  RMSE: {metrics_gb['RMSE']:.4f}°C")
print(f"  MAE:  {metrics_gb['MAE']:.4f}°C")
print(f"  R²:   {metrics_gb['R² Score']:.4f}")

# ============================================================================
# OVERFITTING DETECTION: TRAINING VS TESTING PERFORMANCE
# ============================================================================
print("\n" + "="*70)
print("OVERFITTING DETECTION: TRAINING VS TESTING PERFORMANCE")
print("="*70)

"""
Academic Reasoning for Overfitting Detection:
---------------------------------------------
Overfitting occurs when a model performs well on training data but poorly on 
testing data. This indicates the model has memorized training patterns rather 
than learning generalizable relationships.

Key Indicators:
- Large gap between training and testing R² scores
- Training RMSE/MAE much lower than testing RMSE/MAE
- Model complexity too high relative to data size

Overfitting Control:
- Regularization parameters (max_depth, min_samples_split, learning_rate)
- Time-based split (prevents data leakage)
- Multiple models comparison (identifies overfitted models)
"""

print("\nTraining vs Testing R² Comparison:")
print("-" * 50)
print(f"Linear Regression:")
print(f"  Training R²: {metrics_lr_train['R² Score']:.4f}")
print(f"  Testing R²:  {metrics_lr['R² Score']:.4f}")
print(f"  Gap: {abs(metrics_lr_train['R² Score'] - metrics_lr['R² Score']):.4f}")

print(f"\nRandom Forest:")
print(f"  Training R²: {metrics_rf_train['R² Score']:.4f}")
print(f"  Testing R²:  {metrics_rf['R² Score']:.4f}")
print(f"  Gap: {abs(metrics_rf_train['R² Score'] - metrics_rf['R² Score']):.4f}")

print(f"\nGradient Boosting:")
print(f"  Training R²: {metrics_gb_train['R² Score']:.4f}")
print(f"  Testing R²:  {metrics_gb['R² Score']:.4f}")
print(f"  Gap: {abs(metrics_gb_train['R² Score'] - metrics_gb['R² Score']):.4f}")

# Check for overfitting (gap > 0.1 indicates potential overfitting)
print("\nOverfitting Assessment:")
print("-" * 50)
for model_name, train_r2, test_r2 in [
    ('Linear Regression', metrics_lr_train['R² Score'], metrics_lr['R² Score']),
    ('Random Forest', metrics_rf_train['R² Score'], metrics_rf['R² Score']),
    ('Gradient Boosting', metrics_gb_train['R² Score'], metrics_gb['R² Score'])
]:
    gap = abs(train_r2 - test_r2)
    if gap > 0.15:
        status = "⚠ Potential overfitting (large gap)"
    elif gap > 0.1:
        status = "⚠ Moderate overfitting risk"
    else:
        status = "✓ Good generalization"
    print(f"{model_name}: {status} (gap: {gap:.4f})")

# ============================================================================
# STEP 4: CREATE COMPARISON TABLE
# ============================================================================
print("\n" + "="*70)
print("STEP 4: MODEL COMPARISON TABLE")
print("="*70)

# Combine all metrics into DataFrame
comparison_df = pd.DataFrame([metrics_lr, metrics_rf, metrics_gb])

print("\nModel Comparison Table:")
print("=" * 70)
print(comparison_df.to_string(index=False))
print("=" * 70)

# Save comparison table
comparison_df.to_csv('outputs/csv/model_comparison_results.csv', index=False)
print("\n✓ Comparison table saved: outputs/csv/model_comparison_results.csv")

# ============================================================================
# STEP 5: IDENTIFY BEST MODEL
# ============================================================================
print("\n" + "="*70)
print("STEP 5: BEST MODEL SELECTION")
print("="*70)

"""
Academic Approach to Model Selection:
-------------------------------------
1. Lower RMSE and MAE are better (closer to zero)
2. Higher R² is better (closer to 1)
3. Consider multiple metrics - no single metric is perfect
4. Overfitting control: Prefer models with small training-testing gap
5. Why accuracy alone is not enough:
   - High accuracy on training but low on testing = overfitting
   - Need to balance accuracy with generalization
   - Interpretability matters for scientific understanding
   - Model complexity vs. data size trade-off
6. With monthly data (132 observations), we have sufficient samples for 
   robust evaluation compared to yearly data
"""

# Find best model for each metric
best_rmse = comparison_df.loc[comparison_df['RMSE'].idxmin(), 'Model']
best_mae = comparison_df.loc[comparison_df['MAE'].idxmin(), 'Model']
best_r2 = comparison_df.loc[comparison_df['R² Score'].idxmax(), 'Model']

print("\nBest model by metric:")
print(f"  Lowest RMSE:  {best_rmse}")
print(f"  Lowest MAE:   {best_mae}")
print(f"  Highest R²:   {best_r2}")

# Overall best model (composite score)
# We'll use a simple ranking system: rank each metric, sum ranks
comparison_df['RMSE_Rank'] = comparison_df['RMSE'].rank()
comparison_df['MAE_Rank'] = comparison_df['MAE'].rank()
comparison_df['R2_Rank'] = comparison_df['R² Score'].rank(ascending=False)  # Higher is better
comparison_df['Total_Rank'] = comparison_df['RMSE_Rank'] + comparison_df['MAE_Rank'] + comparison_df['R2_Rank']

best_model_overall = comparison_df.loc[comparison_df['Total_Rank'].idxmin(), 'Model']

print(f"\n{'='*70}")
print(f"OVERALL BEST MODEL: {best_model_overall}")
print(f"{'='*70}")

# Save best model indicator
best_model_info = {
    'Best_Model': best_model_overall,
    'RMSE': comparison_df[comparison_df['Model'] == best_model_overall]['RMSE'].values[0],
    'MAE': comparison_df[comparison_df['Model'] == best_model_overall]['MAE'].values[0],
    'R²': comparison_df[comparison_df['Model'] == best_model_overall]['R² Score'].values[0]
}

best_model_df = pd.DataFrame([best_model_info])
best_model_df.to_csv('outputs/csv/best_model_info.csv', index=False)
print(f"\n✓ Best model info saved: outputs/csv/best_model_info.csv")

# ============================================================================
# STEP 6: VISUALIZATION - ACTUAL VS PREDICTED
# ============================================================================
print("\n" + "="*70)
print("STEP 6: VISUALIZATION - ACTUAL VS PREDICTED")
print("="*70)

"""
Academic Reasoning:
- Actual vs Predicted plots show model performance visually
- Points close to diagonal line indicate good predictions
- Patterns in residuals may indicate model limitations
"""

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

models_info = [
    (y_test, y_test_pred_lr, 'Linear Regression', metrics_lr),
    (y_test, y_test_pred_rf, 'Random Forest', metrics_rf),
    (y_test, y_test_pred_gb, 'Gradient Boosting', metrics_gb)
]

for idx, (y_true, y_pred, model_name, metrics) in enumerate(models_info):
    ax = axes[idx]
    
    # Scatter plot
    ax.scatter(y_true, y_pred, s=100, alpha=0.7, edgecolors='black', linewidth=1.5)
    
    # Perfect prediction line (y=x)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    # Labels and title
    ax.set_xlabel('Actual Temperature (°C)', fontweight='bold')
    ax.set_ylabel('Predicted Temperature (°C)', fontweight='bold')
    ax.set_title(f'{model_name}\nRMSE: {metrics["RMSE"]:.3f}°C | R²: {metrics["R² Score"]:.3f}', 
                 fontweight='bold', pad=10)
    ax.grid(True, alpha=0.3)
    ax.legend()

plt.suptitle('Model Performance: Actual vs Predicted Temperature', 
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('outputs/png/05_model_comparison_actual_vs_predicted.png', dpi=300, bbox_inches='tight')
print("\n✓ Plot saved: 05_model_comparison_actual_vs_predicted.png")
plt.close()

# ============================================================================
# STEP 7: VISUALIZATION - METRICS COMPARISON
# ============================================================================
print("\n" + "="*70)
print("STEP 7: VISUALIZATION - METRICS COMPARISON")
print("="*70)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# RMSE comparison
axes[0].bar(comparison_df['Model'], comparison_df['RMSE'], 
            color=['#FF6B6B', '#4ECDC4', '#45B7D1'], alpha=0.7, edgecolor='black')
axes[0].set_ylabel('RMSE (°C)', fontweight='bold')
axes[0].set_title('RMSE Comparison\n(Lower is Better)', fontweight='bold')
axes[0].tick_params(axis='x', rotation=45)
axes[0].grid(True, alpha=0.3, axis='y')

# MAE comparison
axes[1].bar(comparison_df['Model'], comparison_df['MAE'], 
            color=['#FF6B6B', '#4ECDC4', '#45B7D1'], alpha=0.7, edgecolor='black')
axes[1].set_ylabel('MAE (°C)', fontweight='bold')
axes[1].set_title('MAE Comparison\n(Lower is Better)', fontweight='bold')
axes[1].tick_params(axis='x', rotation=45)
axes[1].grid(True, alpha=0.3, axis='y')

# R² comparison
axes[2].bar(comparison_df['Model'], comparison_df['R² Score'], 
            color=['#FF6B6B', '#4ECDC4', '#45B7D1'], alpha=0.7, edgecolor='black')
axes[2].set_ylabel('R² Score', fontweight='bold')
axes[2].set_title('R² Score Comparison\n(Higher is Better)', fontweight='bold')
axes[2].tick_params(axis='x', rotation=45)
axes[2].grid(True, alpha=0.3, axis='y')

plt.suptitle('Model Evaluation Metrics Comparison', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/png/06_metrics_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Plot saved: 06_metrics_comparison.png")
plt.close()

# ============================================================================
# STEP 8: RESIDUAL ANALYSIS
# ============================================================================
print("\n" + "="*70)
print("STEP 8: RESIDUAL ANALYSIS")
print("="*70)

"""
Academic Reasoning:
- Residuals = Actual - Predicted
- Residual plots help identify:
  * Systematic biases (non-random patterns)
  * Heteroscedasticity (varying error variance)
  * Outliers
- Good models have random, centered residuals
"""

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, (y_true, y_pred, model_name, metrics) in enumerate(models_info):
    ax = axes[idx]
    residuals = y_true - y_pred
    
    # Residual plot
    ax.scatter(y_pred, residuals, s=100, alpha=0.7, edgecolors='black', linewidth=1.5)
    ax.axhline(y=0, color='r', linestyle='--', linewidth=2, label='Zero Error')
    ax.set_xlabel('Predicted Temperature (°C)', fontweight='bold')
    ax.set_ylabel('Residuals (Actual - Predicted)', fontweight='bold')
    ax.set_title(f'{model_name}\nResidual Analysis', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()

plt.suptitle('Residual Analysis: Checking Model Assumptions', 
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/png/07_residual_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Plot saved: 07_residual_analysis.png")
plt.close()

# ============================================================================
# STEP 9: ACADEMIC INTERPRETATION GUIDE
# ============================================================================
print("\n" + "="*70)
print("STEP 9: ACADEMIC MODEL SELECTION GUIDE")
print("="*70)

print("""
HOW TO CHOOSE THE BEST MODEL (ACADEMIC APPROACH):
--------------------------------------------------

1. METRIC INTERPRETATION:
   - RMSE < 1.0°C: Excellent (predictions within 1°C on average)
   - RMSE 1.0-2.0°C: Good
   - RMSE > 2.0°C: May need improvement
   
   - R² > 0.9: Excellent explanatory power
   - R² 0.7-0.9: Good
   - R² < 0.7: Moderate (consider feature engineering or more data)

2. CONSIDER MULTIPLE METRICS:
   - Don't rely on a single metric
   - Balance between RMSE, MAE, and R²
   - Consider model interpretability (Linear Regression wins here)

3. MODEL COMPLEXITY TRADE-OFF:
   - Simpler models (Linear) are more interpretable
   - Complex models (RF, GB) may overfit with small datasets
   - For small datasets, simpler models often generalize better

4. DOMAIN CONSIDERATIONS:
   - If interpretability is crucial: Choose Linear Regression
   - If prediction accuracy is paramount: Choose best-performing model
   - If feature importance matters: Choose tree-based models

5. VALIDATION:
   - With limited data, results should be interpreted cautiously
   - Consider cross-validation for more robust evaluation
   - Be aware of overfitting risks with small sample sizes
""")

print("\n" + "="*70)
print("MODEL EVALUATION COMPLETE")
print("="*70)
print(f"\nBest Model Selected: {best_model_overall}")
print("Next step: Feature importance and urbanization interpretation (Phase 5)")

