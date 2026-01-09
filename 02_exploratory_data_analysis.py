"""
PHASE 2: EXPLORATORY DATA ANALYSIS (EDA)
==========================================

This script performs comprehensive exploratory data analysis to understand
the relationships between urbanization (proxied by monthly nightlights) and 
monthly temperature patterns.

Academic Rationale:
- EDA is essential for understanding data distributions and relationships
- Visualizations help identify trends, patterns, and potential outliers
- Correlation analysis quantifies the strength and direction of relationships
- Monthly resolution reveals seasonal patterns and short-term associations
- This analysis supports the hypothesis that human activity (nightlights) 
  influences local climate patterns

Step-by-step Process:
1. Load preprocessed monthly data
2. Create visualizations (Year-Month vs variables, relationships)
3. Compute correlation coefficients (Pearson and Spearman)
4. Interpret results in context of urbanization/human-activity influence
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
import os

# Set style for academic-quality plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 11

# Create output directories if they don't exist
os.makedirs('outputs/csv', exist_ok=True)
os.makedirs('outputs/png', exist_ok=True)

# ============================================================================
# STEP 1: LOAD PREPROCESSED DATA
# ============================================================================
print("="*70)
print("PHASE 2: EXPLORATORY DATA ANALYSIS")
print("="*70)

print("\nLoading preprocessed monthly dataset...")
df = pd.read_csv('outputs/csv/Chennai_Monthly_Aggregated.csv')
print(f"Dataset shape: {df.shape}")
print(f"\nFirst few rows:")
print(df.head())

# Create Year-Month string for plotting
df['YearMonth'] = df['Year'].astype(str) + '-' + df['Month'].astype(str).str.zfill(2)
df['YearMonthNum'] = df['Year'] + (df['Month'] - 1) / 12  # For continuous x-axis

# ============================================================================
# STEP 2: TEMPORAL TREND ANALYSIS - YEAR-MONTH VS TEMPERATURE
# ============================================================================
print("\n" + "="*70)
print("STEP 2: TEMPORAL TREND ANALYSIS - MONTHLY TEMPERATURE TREND")
print("="*70)

"""
Academic Reasoning:
- Monthly temperature trends show seasonal patterns and long-term changes
- This visualization reveals both intra-annual (seasonal) and inter-annual 
  (long-term) temperature patterns
- Helps identify whether temperature is increasing over time and how it varies 
  seasonally
"""

fig, ax = plt.subplots(figsize=(16, 6))

# Plot monthly temperature over time
ax.plot(df['YearMonthNum'], df['Avg_Monthly_Temperature'], 
        linewidth=1.5, alpha=0.7, color='#3498db', label='Monthly Temperature')

# Add trend line
z = np.polyfit(df['YearMonthNum'], df['Avg_Monthly_Temperature'], 1)
p = np.poly1d(z)
ax.plot(df['YearMonthNum'], p(df['YearMonthNum']), "r--", 
        linewidth=2, alpha=0.8, label=f'Trend: y={z[0]:.3f}x+{z[1]:.2f}')

ax.set_xlabel('Year', fontsize=12, fontweight='bold')
ax.set_ylabel('Average Monthly Temperature (°C)', fontsize=12, fontweight='bold')
ax.set_title('Monthly Temperature Trend in Chennai (2014-2024)\nAverage Monthly Temperature Over Time', 
             fontsize=14, fontweight='bold', pad=15)
ax.grid(True, alpha=0.3, linestyle='--')
ax.legend(fontsize=11)

# Format x-axis to show years
years = df['Year'].unique()
ax.set_xticks([y for y in years])
ax.set_xticklabels([str(int(y)) for y in years], rotation=45)

plt.tight_layout()
plt.savefig('outputs/png/01_monthly_temperature_trend.png', dpi=300, bbox_inches='tight')
print("\n✓ Plot saved: outputs/png/01_monthly_temperature_trend.png")
plt.close()

# ============================================================================
# STEP 3: TEMPORAL TREND ANALYSIS - YEAR-MONTH VS NIGHTLIGHT
# ============================================================================
print("\n" + "="*70)
print("STEP 3: TEMPORAL TREND ANALYSIS - MONTHLY URBANIZATION TREND")
print("="*70)

"""
Academic Reasoning:
- Monthly nightlight intensity shows short-term human activity patterns
- Urbanization influence is inferred through learned patterns over time
- This visualization demonstrates how human activity (proxied by nightlights) 
  varies monthly and over years
"""

fig, ax = plt.subplots(figsize=(16, 6))

# Plot monthly nightlight over time
ax.plot(df['YearMonthNum'], df['NightLight'], 
        linewidth=1.5, alpha=0.7, color='#2c3e50', label='Monthly Nightlight Intensity')

# Add trend line
z = np.polyfit(df['YearMonthNum'], df['NightLight'], 1)
p = np.poly1d(z)
ax.plot(df['YearMonthNum'], p(df['YearMonthNum']), "r--", 
        linewidth=2, alpha=0.8, label=f'Trend: y={z[0]:.3f}x+{z[1]:.2f}')

ax.set_xlabel('Year', fontsize=12, fontweight='bold')
ax.set_ylabel('VIIRS Nightlight Intensity', fontsize=12, fontweight='bold')
ax.set_title('Monthly Urbanization Trend in Chennai (2014-2024)\nVIIRS Nighttime Light Intensity Over Time', 
             fontsize=14, fontweight='bold', pad=15)
ax.grid(True, alpha=0.3, linestyle='--')
ax.legend(fontsize=11)

# Format x-axis
ax.set_xticks([y for y in years])
ax.set_xticklabels([str(int(y)) for y in years], rotation=45)

plt.tight_layout()
plt.savefig('outputs/png/02_monthly_nightlight_trend.png', dpi=300, bbox_inches='tight')
print("✓ Plot saved: outputs/png/02_monthly_nightlight_trend.png")
plt.close()

# ============================================================================
# STEP 4: RELATIONSHIP ANALYSIS - NIGHTLIGHT VS TEMPERATURE
# ============================================================================
print("\n" + "="*70)
print("STEP 4: URBANIZATION-TEMPERATURE RELATIONSHIP")
print("="*70)

"""
Academic Reasoning:
- This is the core relationship of interest
- Positive correlation suggests that as human activity increases (higher nightlights),
  temperature also increases
- This supports the urban heat island (UHI) effect hypothesis
- Monthly resolution captures both seasonal and activity-driven patterns
"""

fig, ax = plt.subplots(figsize=(12, 8))

# Scatter plot with color coding by year
scatter = ax.scatter(df['NightLight'], df['Avg_Monthly_Temperature'], 
                     s=50, alpha=0.6, c=df['Year'], 
                     cmap='viridis', edgecolors='black', linewidth=0.5)

# Add regression line
z = np.polyfit(df['NightLight'], df['Avg_Monthly_Temperature'], 1)
p = np.poly1d(z)
ax.plot(df['NightLight'], p(df['NightLight']), "r-", linewidth=2, 
        label=f'Linear Fit: y={z[0]:.3f}x+{z[1]:.2f}')

# Colorbar
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Year', fontsize=11, fontweight='bold')

ax.set_xlabel('VIIRS Nightlight Intensity (Human Activity Proxy)', 
              fontsize=12, fontweight='bold')
ax.set_ylabel('Average Monthly Temperature (°C)', fontsize=12, fontweight='bold')
ax.set_title('Urbanization Impact on Local Temperature\nMonthly Nightlight Intensity vs Average Monthly Temperature', 
             fontsize=14, fontweight='bold', pad=15)
ax.grid(True, alpha=0.3, linestyle='--')
ax.legend(fontsize=11)

plt.tight_layout()
plt.savefig('outputs/png/03_nightlight_vs_temperature_monthly.png', dpi=300, bbox_inches='tight')
print("✓ Plot saved: outputs/png/03_nightlight_vs_temperature_monthly.png")
plt.close()

# ============================================================================
# STEP 5: SEASONAL PATTERN ANALYSIS
# ============================================================================
print("\n" + "="*70)
print("STEP 5: SEASONAL PATTERN ANALYSIS")
print("="*70)

"""
Academic Reasoning:
- Monthly data reveals seasonal patterns in both temperature and human activity
- Understanding seasonal variations helps interpret the urbanization-temperature 
  relationship
- Some months may show stronger associations than others
"""

fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Temperature by month
monthly_avg_temp = df.groupby('Month')['Avg_Monthly_Temperature'].mean()
axes[0].bar(monthly_avg_temp.index, monthly_avg_temp.values, 
            color='#3498db', alpha=0.7, edgecolor='black')
axes[0].set_xlabel('Month', fontweight='bold')
axes[0].set_ylabel('Average Temperature (°C)', fontweight='bold')
axes[0].set_title('A) Average Monthly Temperature by Month (2014-2024)', 
                 fontweight='bold')
axes[0].set_xticks(range(1, 13))
axes[0].set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                         'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
axes[0].grid(True, alpha=0.3, axis='y')

# Nightlight by month
monthly_avg_nl = df.groupby('Month')['NightLight'].mean()
axes[1].bar(monthly_avg_nl.index, monthly_avg_nl.values, 
            color='#2c3e50', alpha=0.7, edgecolor='black')
axes[1].set_xlabel('Month', fontweight='bold')
axes[1].set_ylabel('Average Nightlight Intensity', fontweight='bold')
axes[1].set_title('B) Average Monthly Nightlight Intensity by Month (2014-2024)', 
                 fontweight='bold')
axes[1].set_xticks(range(1, 13))
axes[1].set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                         'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
axes[1].grid(True, alpha=0.3, axis='y')

plt.suptitle('Seasonal Patterns in Temperature and Human Activity', 
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('outputs/png/04_seasonal_patterns.png', dpi=300, bbox_inches='tight')
print("✓ Plot saved: outputs/png/04_seasonal_patterns.png")
plt.close()

# ============================================================================
# STEP 6: COMPREHENSIVE VISUALIZATION DASHBOARD
# ============================================================================
print("\n" + "="*70)
print("STEP 6: COMPREHENSIVE VISUALIZATION DASHBOARD")
print("="*70)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Panel 1: Temperature trend
axes[0, 0].plot(df['YearMonthNum'], df['Avg_Monthly_Temperature'], 
                linewidth=1.5, color='#3498db', alpha=0.7)
axes[0, 0].set_xlabel('Year', fontweight='bold')
axes[0, 0].set_ylabel('Temperature (°C)', fontweight='bold')
axes[0, 0].set_title('A) Monthly Temperature Trend', fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_xticks([y for y in years])
axes[0, 0].set_xticklabels([str(int(y)) for y in years], rotation=45)

# Panel 2: Nightlight trend
axes[0, 1].plot(df['YearMonthNum'], df['NightLight'], 
                linewidth=1.5, color='#2c3e50', alpha=0.7)
axes[0, 1].set_xlabel('Year', fontweight='bold')
axes[0, 1].set_ylabel('Nightlight Intensity', fontweight='bold')
axes[0, 1].set_title('B) Monthly Nightlight Trend', fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].set_xticks([y for y in years])
axes[0, 1].set_xticklabels([str(int(y)) for y in years], rotation=45)

# Panel 3: Scatter plot
scatter = axes[1, 0].scatter(df['NightLight'], df['Avg_Monthly_Temperature'], 
                             s=30, alpha=0.6, c=df['Year'], cmap='viridis')
z = np.polyfit(df['NightLight'], df['Avg_Monthly_Temperature'], 1)
p = np.poly1d(z)
axes[1, 0].plot(df['NightLight'], p(df['NightLight']), "r-", linewidth=2)
axes[1, 0].set_xlabel('Nightlight Intensity', fontweight='bold')
axes[1, 0].set_ylabel('Temperature (°C)', fontweight='bold')
axes[1, 0].set_title('C) Nightlight vs Temperature', fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)
plt.colorbar(scatter, ax=axes[1, 0], label='Year')

# Panel 4: Correlation heatmap
corr_data = df[['Year', 'Month', 'NightLight', 'Avg_Monthly_Temperature']].corr()
im = axes[1, 1].imshow(corr_data, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
axes[1, 1].set_xticks(range(len(corr_data.columns)))
axes[1, 1].set_yticks(range(len(corr_data.columns)))
axes[1, 1].set_xticklabels(corr_data.columns, rotation=45, ha='right')
axes[1, 1].set_yticklabels(corr_data.columns)
axes[1, 1].set_title('D) Correlation Matrix', fontweight='bold')

# Add correlation values
for i in range(len(corr_data.columns)):
    for j in range(len(corr_data.columns)):
        text = axes[1, 1].text(j, i, f'{corr_data.iloc[i, j]:.2f}',
                              ha="center", va="center", color="black", fontweight='bold')

plt.colorbar(im, ax=axes[1, 1])
plt.suptitle('Exploratory Data Analysis: Monthly Urbanization and Temperature Relationships', 
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('outputs/png/05_comprehensive_eda_dashboard.png', dpi=300, bbox_inches='tight')
print("✓ Plot saved: outputs/png/05_comprehensive_eda_dashboard.png")
plt.close()

# ============================================================================
# STEP 7: CORRELATION ANALYSIS
# ============================================================================
print("\n" + "="*70)
print("STEP 7: QUANTITATIVE CORRELATION ANALYSIS")
print("="*70)

"""
Academic Reasoning:
- Pearson correlation measures linear relationships (parametric)
- Spearman correlation measures monotonic relationships (non-parametric, rank-based)
- Both provide different insights into the strength of relationships
- Statistical significance testing validates whether correlations are meaningful
"""

print("\n7.1 Pearson Correlation (Linear Relationship):")
print("-" * 50)
print("Pearson correlation assumes linearity and normal distribution.")
print("It measures how well a linear equation can describe the relationship.\n")

# Year vs Temperature
pearson_year_temp, p_year_temp = pearsonr(df['Year'], df['Avg_Monthly_Temperature'])
print(f"Year vs Temperature:")
print(f"  Correlation: {pearson_year_temp:.4f}")
print(f"  p-value: {p_year_temp:.4f}")
print(f"  Interpretation: {'Strong' if abs(pearson_year_temp) > 0.7 else 'Moderate' if abs(pearson_year_temp) > 0.3 else 'Weak'} {'positive' if pearson_year_temp > 0 else 'negative'} correlation")
print(f"  Significance: {'Significant' if p_year_temp < 0.05 else 'Not significant'} at α=0.05\n")

# Month vs Temperature (seasonal)
pearson_month_temp, p_month_temp = pearsonr(df['Month'], df['Avg_Monthly_Temperature'])
print(f"Month vs Temperature (Seasonal):")
print(f"  Correlation: {pearson_month_temp:.4f}")
print(f"  p-value: {p_month_temp:.4f}")
print(f"  Interpretation: {'Strong' if abs(pearson_month_temp) > 0.7 else 'Moderate' if abs(pearson_month_temp) > 0.3 else 'Weak'} {'positive' if pearson_month_temp > 0 else 'negative'} correlation")
print(f"  Significance: {'Significant' if p_month_temp < 0.05 else 'Not significant'} at α=0.05\n")

# NightLight vs Temperature (MOST IMPORTANT)
pearson_nl_temp, p_nl_temp = pearsonr(df['NightLight'], df['Avg_Monthly_Temperature'])
print(f"NightLight vs Temperature (KEY RELATIONSHIP):")
print(f"  Correlation: {pearson_nl_temp:.4f}")
print(f"  p-value: {p_nl_temp:.4f}")
print(f"  Interpretation: {'Strong' if abs(pearson_nl_temp) > 0.7 else 'Moderate' if abs(pearson_nl_temp) > 0.3 else 'Weak'} {'positive' if pearson_nl_temp > 0 else 'negative'} correlation")
print(f"  Significance: {'Significant' if p_nl_temp < 0.05 else 'Not significant'} at α=0.05\n")

print("\n7.2 Spearman Correlation (Monotonic Relationship):")
print("-" * 50)
print("Spearman correlation measures monotonic relationships without assuming")
print("linearity. It's more robust to outliers and non-normal distributions.\n")

# Year vs Temperature
spearman_year_temp, p_s_year_temp = spearmanr(df['Year'], df['Avg_Monthly_Temperature'])
print(f"Year vs Temperature:")
print(f"  Correlation: {spearman_year_temp:.4f}")
print(f"  p-value: {p_s_year_temp:.4f}\n")

# Month vs Temperature
spearman_month_temp, p_s_month_temp = spearmanr(df['Month'], df['Avg_Monthly_Temperature'])
print(f"Month vs Temperature:")
print(f"  Correlation: {spearman_month_temp:.4f}")
print(f"  p-value: {p_s_month_temp:.4f}\n")

# NightLight vs Temperature
spearman_nl_temp, p_s_nl_temp = spearmanr(df['NightLight'], df['Avg_Monthly_Temperature'])
print(f"NightLight vs Temperature (KEY RELATIONSHIP):")
print(f"  Correlation: {spearman_nl_temp:.4f}")
print(f"  p-value: {p_s_nl_temp:.4f}\n")

# ============================================================================
# STEP 8: SUMMARY STATISTICS
# ============================================================================
print("\n" + "="*70)
print("STEP 8: SUMMARY STATISTICS")
print("="*70)

print("\nDescriptive Statistics:")
print("-" * 50)
print(df[['Year', 'Month', 'NightLight', 'Avg_Monthly_Temperature']].describe())

# ============================================================================
# STEP 9: EXPORT CORRELATION RESULTS
# ============================================================================
print("\n" + "="*70)
print("STEP 9: EXPORTING CORRELATION RESULTS")
print("="*70)

correlation_results = {
    'Relationship': ['Year vs Temperature', 'Month vs Temperature', 'NightLight vs Temperature'],
    'Pearson_Correlation': [pearson_year_temp, pearson_month_temp, pearson_nl_temp],
    'Pearson_p_value': [p_year_temp, p_month_temp, p_nl_temp],
    'Spearman_Correlation': [spearman_year_temp, spearman_month_temp, spearman_nl_temp],
    'Spearman_p_value': [p_s_year_temp, p_s_month_temp, p_s_nl_temp]
}

corr_df = pd.DataFrame(correlation_results)
corr_df.to_csv('outputs/csv/correlation_analysis_results.csv', index=False)
print("\n✓ Correlation results exported to: outputs/csv/correlation_analysis_results.csv")
print(corr_df)

print("\n" + "="*70)
print("EXPLORATORY DATA ANALYSIS COMPLETE")
print("="*70)
