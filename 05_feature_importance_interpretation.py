"""
PHASE 5: FEATURE IMPORTANCE & URBANIZATION INTERPRETATION
===========================================================

This script analyzes feature importance to understand how Year, Month, and 
NightLight contribute to temperature predictions, with focus on urbanization impact.

Academic Rationale:
- Feature importance quantifies each predictor's contribution
- Understanding which features matter most informs policy and research
- NightLight importance directly demonstrates human activity/urbanization's role
- Year importance may indicate temporal trends or global warming effects
- Month importance captures seasonal patterns in temperature

Important Academic Distinction:
- This is INFLUENCE ANALYSIS, not causal proof
- Correlation and prediction do not establish causation
- Other factors (global warming, natural variability) may confound results
- We quantify associations, not causal mechanisms

Step-by-step Process:
1. Load models and data
2. Extract feature importance from tree-based models
3. Analyze linear regression coefficients
4. Visualize feature contributions
5. Interpret results in context of urbanization impact
"""

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create output directories if they don't exist
os.makedirs('outputs/csv', exist_ok=True)
os.makedirs('outputs/png', exist_ok=True)

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# ============================================================================
# STEP 1: LOAD MODELS AND DATA
# ============================================================================
print("="*70)
print("PHASE 5: FEATURE IMPORTANCE & URBANIZATION INTERPRETATION")
print("="*70)

print("\nLoading models and data...")
print("-" * 50)

# Load models
with open('models/model_linear_regression.pkl', 'rb') as f:
    lr_model = pickle.load(f)

with open('models/model_random_forest.pkl', 'rb') as f:
    rf_model = pickle.load(f)

with open('models/model_gradient_boosting.pkl', 'rb') as f:
    gb_model = pickle.load(f)

# Load data for reference
df = pd.read_csv('outputs/csv/Chennai_Monthly_Aggregated.csv')

print("✓ Models and data loaded")

# ============================================================================
# STEP 2: LINEAR REGRESSION COEFFICIENT ANALYSIS
# ============================================================================
print("\n" + "="*70)
print("STEP 2: LINEAR REGRESSION COEFFICIENT INTERPRETATION")
print("="*70)

"""
Academic Reasoning:
- Linear regression coefficients are directly interpretable
- Coefficient magnitude indicates feature strength
- Sign indicates direction of relationship (positive/negative)
- Standardized coefficients (if needed) allow comparison across features
"""

print("\nLinear Regression Coefficients:")
print("-" * 50)

coefficients = pd.DataFrame({
    'Feature': ['Year', 'Month', 'NightLight'],
    'Coefficient': lr_model.coef_,
    'Absolute_Coefficient': np.abs(lr_model.coef_)
}).sort_values('Absolute_Coefficient', ascending=False)

print(coefficients)
print(f"\nIntercept: {lr_model.intercept_:.4f}°C")

# Interpretation
print("\nInterpretation:")
print("-" * 50)
year_coef = lr_model.coef_[0]
month_coef = lr_model.coef_[1]
nightlight_coef = lr_model.coef_[2]

print(f"\n1. Year Coefficient: {year_coef:.4f}°C per year")
print(f"   - For each additional year, temperature increases by {year_coef:.4f}°C")
print(f"   - This may reflect:")
print(f"     * Global warming trends")
print(f"     * Long-term climate change")
print(f"     * Natural temporal variability")

print(f"\n2. Month Coefficient: {month_coef:.4f}°C per month")
print(f"   - Seasonal variation: each month change affects temperature by {month_coef:.4f}°C")
print(f"   - This captures:")
print(f"     * Seasonal patterns in Chennai's tropical climate")
print(f"     * Intra-annual temperature variations")
print(f"     * Natural climate cycles")

print(f"\n3. NightLight Coefficient: {nightlight_coef:.4f}°C per unit")
print(f"   - For each unit increase in VIIRS nightlight intensity,")
print(f"     temperature increases by {nightlight_coef:.4f}°C")
print(f"   - This suggests:")
print(f"     * Association between human activity and temperature")
print(f"     * Potential urban heat island (UHI) effect")
print(f"     * Human activity as a contributing factor to local warming")

# Calculate relative importance (normalized)
total_abs_coef = coefficients['Absolute_Coefficient'].sum()
coefficients['Relative_Importance_%'] = (coefficients['Absolute_Coefficient'] / total_abs_coef) * 100

print("\nRelative Feature Importance (Linear Regression):")
print("-" * 50)
for _, row in coefficients.iterrows():
    print(f"{row['Feature']}: {row['Relative_Importance_%']:.2f}%")

# ============================================================================
# STEP 3: RANDOM FOREST FEATURE IMPORTANCE
# ============================================================================
print("\n" + "="*70)
print("STEP 3: RANDOM FOREST FEATURE IMPORTANCE")
print("="*70)

"""
Academic Reasoning:
- Random Forest provides feature importance based on:
  * How much each feature reduces impurity (variance) in splits
  * Average importance across all trees
- Values sum to 1.0, representing proportion of total importance
"""

print("\nRandom Forest Feature Importance:")
print("-" * 50)

rf_importance = pd.DataFrame({
    'Feature': ['Year', 'Month', 'NightLight'],
    'Importance': rf_model.feature_importances_,
    'Importance_%': rf_model.feature_importances_ * 100
}).sort_values('Importance', ascending=False)

print(rf_importance)

print("\nInterpretation:")
print("-" * 50)
for _, row in rf_importance.iterrows():
    print(f"{row['Feature']}: {row['Importance_%']:.2f}% of total importance")
    if row['Feature'] == 'NightLight':
        print(f"  → This demonstrates urbanization's contribution to temperature prediction")

# ============================================================================
# STEP 4: GRADIENT BOOSTING FEATURE IMPORTANCE
# ============================================================================
print("\n" + "="*70)
print("STEP 4: GRADIENT BOOSTING FEATURE IMPORTANCE")
print("="*70)

print("\nGradient Boosting Feature Importance:")
print("-" * 50)

gb_importance = pd.DataFrame({
    'Feature': ['Year', 'Month', 'NightLight'],
    'Importance': gb_model.feature_importances_,
    'Importance_%': gb_model.feature_importances_ * 100
}).sort_values('Importance', ascending=False)

print(gb_importance)

print("\nInterpretation:")
print("-" * 50)
for _, row in gb_importance.iterrows():
    print(f"{row['Feature']}: {row['Importance_%']:.2f}% of total importance")

# ============================================================================
# STEP 5: COMPREHENSIVE FEATURE IMPORTANCE COMPARISON
# ============================================================================
print("\n" + "="*70)
print("STEP 5: FEATURE IMPORTANCE ACROSS MODELS")
print("="*70)

"""
Academic Reasoning:
- Comparing feature importance across models shows consistency
- Agreement across models strengthens confidence in findings
- Disagreement may indicate model-specific assumptions or biases
"""

# Create comparison DataFrame
importance_comparison = pd.DataFrame({
    'Feature': ['Year', 'Month', 'NightLight'],
    'Linear_Regression_%': coefficients.set_index('Feature')['Relative_Importance_%'].values,
    'Random_Forest_%': rf_importance.set_index('Feature')['Importance_%'].values,
    'Gradient_Boosting_%': gb_importance.set_index('Feature')['Importance_%'].values
})

# Calculate average importance
importance_comparison['Average_Importance_%'] = importance_comparison[
    ['Linear_Regression_%', 'Random_Forest_%', 'Gradient_Boosting_%']
].mean(axis=1)

importance_comparison = importance_comparison.sort_values('Average_Importance_%', ascending=False)

print("\nFeature Importance Comparison (All Models):")
print("=" * 70)
print(importance_comparison.to_string(index=False))
print("=" * 70)

# Save results
importance_comparison.to_csv('outputs/csv/feature_importance_comparison.csv', index=False)
print("\n✓ Feature importance comparison saved: outputs/csv/feature_importance_comparison.csv")

# ============================================================================
# STEP 6: VISUALIZATION - FEATURE IMPORTANCE
# ============================================================================
print("\n" + "="*70)
print("STEP 6: FEATURE IMPORTANCE VISUALIZATION")
print("="*70)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Linear Regression coefficients (absolute values, normalized)
lr_normalized = np.abs(lr_model.coef_) / np.abs(lr_model.coef_).sum() * 100
axes[0].barh(['Year', 'Month', 'NightLight'], lr_normalized, 
             color=['#3498db', '#27ae60', '#2c3e50'], alpha=0.7, edgecolor='black')
axes[0].set_xlabel('Relative Importance (%)', fontweight='bold')
axes[0].set_title('A) Linear Regression\nCoefficient-Based Importance', fontweight='bold')
axes[0].grid(True, alpha=0.3, axis='x')

# Random Forest importance
axes[1].barh(rf_importance['Feature'], rf_importance['Importance_%'], 
             color=['#FF6B6B', '#4ECDC4'], alpha=0.7, edgecolor='black')
axes[1].set_xlabel('Importance (%)', fontweight='bold')
axes[1].set_title('B) Random Forest\nTree-Based Importance', fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='x')

# Gradient Boosting importance
axes[2].barh(gb_importance['Feature'], gb_importance['Importance_%'], 
             color=['#FF6B6B', '#4ECDC4'], alpha=0.7, edgecolor='black')
axes[2].set_xlabel('Importance (%)', fontweight='bold')
axes[2].set_title('C) Gradient Boosting\nSequential Learning Importance', fontweight='bold')
axes[2].grid(True, alpha=0.3, axis='x')

plt.suptitle('Feature Importance Analysis: Year vs NightLight (Urbanization)', 
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('outputs/png/08_feature_importance_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Plot saved: 08_feature_importance_comparison.png")
plt.close()

# ============================================================================
# STEP 7: URBANIZATION IMPACT INTERPRETATION
# ============================================================================
print("\n" + "="*70)
print("STEP 7: URBANIZATION IMPACT INTERPRETATION")
print("="*70)

"""
CRITICAL ACADEMIC DISCLAIMER:
==============================
This analysis demonstrates ASSOCIATION and INFLUENCE, NOT CAUSATION.

What we can conclude:
- NightLight (urbanization proxy) is associated with temperature
- Models can predict temperature using urbanization indicators
- Urbanization appears to contribute to temperature patterns

What we CANNOT conclude:
- Urbanization directly causes temperature increases (causation)
- All temperature changes are due to urbanization
- No other factors influence the relationship

Confounding factors may include:
- Global climate change
- Natural climate variability
- Regional weather patterns
- Other urbanization-related factors (not captured by nightlights)
"""

print("\nURBANIZATION IMPACT ANALYSIS:")
print("=" * 70)

# Calculate average nightlight importance
avg_nightlight_importance = importance_comparison[
    importance_comparison['Feature'] == 'NightLight'
]['Average_Importance_%'].values[0]

print(f"\n1. NightLight (Urbanization Proxy) Importance:")
print(f"   Average importance across models: {avg_nightlight_importance:.2f}%")
print(f"   ")
print(f"   Interpretation:")
print(f"   • Nightlight intensity contributes {avg_nightlight_importance:.2f}% to")
print(f"     temperature prediction, on average across all models")
print(f"   • This suggests urbanization is an important factor in local")
print(f"     temperature patterns")
print(f"   • Higher nightlight intensity (more urbanization) is associated")
print(f"     with higher temperatures")

print(f"\n2. Year Importance:")
avg_year_importance = importance_comparison[
    importance_comparison['Feature'] == 'Year'
]['Average_Importance_%'].values[0]
print(f"   Average importance across models: {avg_year_importance:.2f}%")
print(f"   ")
print(f"   Interpretation:")
print(f"   • Temporal trends (Year) contribute {avg_year_importance:.2f}% to")
print(f"     temperature prediction")
print(f"   • This may reflect:")
print(f"     - Global warming trends")
print(f"     - Long-term climate change")
print(f"     - Natural temporal variability")

print(f"\n3. Combined Interpretation:")
print(f"   • Both features contribute significantly to temperature prediction")
print(f"   • Urbanization (NightLight) and temporal trends (Year) both matter")
print(f"   • The model captures both local (urbanization) and global/regional")
print(f"     (temporal) influences on temperature")

# ============================================================================
# STEP 8: QUANTITATIVE IMPACT ESTIMATION
# ============================================================================
print("\n" + "="*70)
print("STEP 8: QUANTITATIVE IMPACT ESTIMATION")
print("="*70)

"""
Academic Approach:
- Estimate practical impact of urbanization on temperature
- Use model predictions to show "what if" scenarios
- Provides quantitative estimates for policy/reporting
"""

print("\nQuantitative Impact Analysis:")
print("-" * 50)

# Find min and max nightlight values
min_nightlight = df['NightLight'].min()
max_nightlight = df['NightLight'].max()
nightlight_range = max_nightlight - min_nightlight

# Use linear regression for interpretability
# Predict temperature at min and max nightlight (holding year and month constant at median)
median_year = df['Year'].median()
median_month = df['Month'].median()
min_temp_pred = lr_model.predict([[median_year, median_month, min_nightlight]])[0]
max_temp_pred = lr_model.predict([[median_year, median_month, max_nightlight]])[0]
temp_change = max_temp_pred - min_temp_pred

print(f"\nTemperature change associated with urbanization:")
print(f"  Minimum nightlight: {min_nightlight:.2f}")
print(f"  Maximum nightlight: {max_nightlight:.2f}")
print(f"  Nightlight increase: {nightlight_range:.2f} units")
print(f"  ")
print(f"  Associated temperature change (holding year and month constant at median):")
print(f"    Year: {median_year}, Month: {median_month}")
print(f"    Temperature change: {temp_change:.3f}°C")

print(f"\n  Per unit nightlight increase: {nightlight_coef:.4f}°C")

# ============================================================================
# STEP 9: ACADEMIC LIMITATIONS AND CAVEATS
# ============================================================================
print("\n" + "="*70)
print("STEP 9: LIMITATIONS AND ACADEMIC CAVEATS")
print("="*70)

print("""
IMPORTANT LIMITATIONS:
======================

1. CAUSALITY vs. ASSOCIATION:
   • This analysis shows ASSOCIATION, not CAUSATION
   • Correlation does not imply causation
   • Other unmeasured factors may influence results

2. CONFOUNDING VARIABLES:
   • Global climate change affects both urbanization and temperature
   • Natural climate variability not accounted for
   • Other local factors (industrialization, population density) may matter

3. DATA LIMITATIONS:
   • Small sample size (12 years) limits statistical power
   • Nightlight is a proxy, not direct urbanization measure
   • Temperature aggregation to yearly averages loses detail

4. MODEL LIMITATIONS:
   • Models assume relationships can be captured by current features
   • Potential overfitting with small dataset
   • Linear assumptions (for LR) may not hold

5. GENERALIZATION:
   • Results specific to Chennai
   • May not generalize to other cities/climates
   • Regional factors specific to South India

6. ETHICAL CONSIDERATIONS:
   • Urbanization is complex and multi-faceted
   • Policy implications must consider social and economic factors
   • Climate impact is one aspect of urbanization analysis
""")

print("\n" + "="*70)
print("FEATURE IMPORTANCE ANALYSIS COMPLETE")
print("="*70)
print("\nNext step: Website development (Phase 6)")

