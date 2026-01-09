"""
PHASE 1: DATA PREPROCESSING & VALIDATION
==========================================

This script performs comprehensive data validation and preprocessing for the 
urban climate impact analysis project using MONTHLY resolution data.

Academic Rationale:
- Monthly aggregation provides data points (132 months)
- More granular temporal resolution captures seasonal patterns and short-term 
  variations in both temperature and human activity
- Monthly VIIRS data represents short-term human activity patterns, while 
  urbanization influence is inferred through learned patterns over time
- 2014-2024 restriction: VIIRS monthly data availability limitation from 
  satellite sensors

Why Monthly Data Reduces Overfitting:
1. More training samples: 132 monthly observations 
2. Better generalization: Models learn from more diverse temporal patterns
3. Reduced variance: More data points stabilize model parameters
4. Seasonal patterns: Captures intra-annual variations that yearly data misses

Step-by-step Process:
1. Load temperature dataset (Chennai.csv)
2. Convert date to datetime and extract Year, Month
3. Aggregate temperature to monthly averages
4. Filter to 2014-2024 period
5. Load monthly VIIRS nightlight dataset
6. Merge on Year and Month
7. Validate merged dataset
8. Export cleaned dataset
"""

import pandas as pd
import numpy as np
import os

# Create output directories if they don't exist
os.makedirs('outputs/csv', exist_ok=True)
os.makedirs('outputs/png', exist_ok=True)

# ============================================================================
# STEP 1: LOAD TEMPERATURE DATASET
# ============================================================================
print("="*70)
print("PHASE 1: DATA PREPROCESSING & VALIDATION")
print("="*70)
print("\nSTEP 1: LOADING TEMPERATURE DATASET")
print("="*70)

# Load temperature dataset
print("\nLoading temperature dataset...")
temp_df = pd.read_csv('Chennai.csv')
print(f"Temperature data shape: {temp_df.shape}")
print(f"Columns: {temp_df.columns.tolist()}")

# ============================================================================
# STEP 2: VALIDATE TEMPERATURE DATASET
# ============================================================================
print("\n" + "="*70)
print("STEP 2: VALIDATING TEMPERATURE DATASET")
print("="*70)

# 2.1 Check for missing values
print("\n2.1 Checking for missing values:")
print("-" * 50)
missing_temp = temp_df.isnull().sum()
print(missing_temp[missing_temp > 0] if missing_temp.sum() > 0 else "✓ No missing values found")

# 2.2 Check for duplicates
print("\n2.2 Checking for duplicate rows:")
print("-" * 50)
duplicates_temp = temp_df.duplicated().sum()
print(f"Number of duplicate rows: {duplicates_temp}")

# 2.3 Validate data types
print("\n2.3 Validating data types:")
print("-" * 50)
print(temp_df.dtypes)

# 2.4 Check temperature values
print("\n2.4 Checking temperature values:")
print("-" * 50)
if 'temperature_2m' in temp_df.columns:
    print(f"Temperature column: 'temperature_2m'")
    print(f"Min temperature: {temp_df['temperature_2m'].min():.2f}°C")
    print(f"Max temperature: {temp_df['temperature_2m'].max():.2f}°C")
    print(f"Mean temperature: {temp_df['temperature_2m'].mean():.2f}°C")
    print(f"Standard deviation: {temp_df['temperature_2m'].std():.2f}°C")
    
    # Reasonable range check for Chennai
    reasonable_min, reasonable_max = 15, 45
    outliers = temp_df[(temp_df['temperature_2m'] < reasonable_min) | 
                       (temp_df['temperature_2m'] > reasonable_max)]
    if len(outliers) > 0:
        print(f"\nWARNING: Found {len(outliers)} temperature readings outside reasonable range")
    else:
        print("\n✓ Temperature values are within reasonable range for Chennai")

# ============================================================================
# STEP 3: CONVERT DATE AND EXTRACT YEAR, MONTH
# ============================================================================
print("\n" + "="*70)
print("STEP 3: PROCESSING DATE COLUMN")
print("="*70)

if 'date' in temp_df.columns:
    # Convert to datetime
    temp_df['date'] = pd.to_datetime(temp_df['date'])
    
    # Extract Year and Month
    temp_df['Year'] = temp_df['date'].dt.year
    temp_df['Month'] = temp_df['date'].dt.month
    
    print(f"\nDate range: {temp_df['date'].min()} to {temp_df['date'].max()}")
    print(f"Year range: {temp_df['Year'].min()} to {temp_df['Year'].max()}")
    print(f"Month range: {temp_df['Month'].min()} to {temp_df['Month'].max()}")
else:
    raise ValueError("'date' column not found in temperature dataset")

# ============================================================================
# STEP 4: AGGREGATE TEMPERATURE TO MONTHLY AVERAGES
# ============================================================================
print("\n" + "="*70)
print("STEP 4: AGGREGATING TEMPERATURE TO MONTHLY AVERAGES")
print("="*70)

if 'temperature_2m' in temp_df.columns:
    # Group by Year and Month, calculate mean temperature
    monthly_temp = temp_df.groupby(['Year', 'Month'])['temperature_2m'].agg([
        'mean', 'std', 'count'
    ]).reset_index()
    monthly_temp.columns = ['Year', 'Month', 'Avg_Monthly_Temperature', 
                            'Std_Temperature', 'Observation_Count']
    
    print("\nMonthly temperature aggregation summary:")
    print("-" * 50)
    print(f"Total monthly observations: {len(monthly_temp)}")
    print(f"\nFirst few rows:")
    print(monthly_temp.head(10))
    
    print(f"\nYear range in aggregated data: {monthly_temp['Year'].min()} to {monthly_temp['Year'].max()}")
else:
    raise ValueError("'temperature_2m' column not found in temperature dataset")

# ============================================================================
# STEP 5: FILTER TO 2014-2024 PERIOD
# ============================================================================
print("\n" + "="*70)
print("STEP 5: FILTERING TO 2014-2024 PERIOD")
print("="*70)

# Filter to 2014-2024
monthly_temp_filtered = monthly_temp[
    (monthly_temp['Year'] >= 2014) & (monthly_temp['Year'] <= 2024)
].copy()

print(f"\nFiltered monthly temperature data:")
print(f"  Observations: {len(monthly_temp_filtered)}")
print(f"  Year range: {monthly_temp_filtered['Year'].min()} to {monthly_temp_filtered['Year'].max()}")
print(f"  Months per year: ~{len(monthly_temp_filtered) / (monthly_temp_filtered['Year'].max() - monthly_temp_filtered['Year'].min() + 1):.1f}")

# ============================================================================
# STEP 6: LOAD MONTHLY VIIRS NIGHTLIGHT DATASET
# ============================================================================
print("\n" + "="*70)
print("STEP 6: LOADING MONTHLY VIIRS NIGHTLIGHT DATASET")
print("="*70)

# Try to load monthly nightlight file (may have different names)
nightlight_files = [
    'Chennai_VIIRS_Monthly_Nightlights_2012_2024.csv'
]

nightlight_df = None
for file in nightlight_files:
    try:
        nightlight_df = pd.read_csv(file)
        print(f"\n✓ Loaded: {file}")
        break
    except FileNotFoundError:
        continue

if nightlight_df is None:
    raise FileNotFoundError("Monthly VIIRS nightlight file not found. Expected: Chennai_VIIRS_Monthly_Nightlights_2014_2024.csv")

print(f"Nightlight data shape: {nightlight_df.shape}")
print(f"Columns: {nightlight_df.columns.tolist()}")

# Filter to 2014-2024 if needed
if 'Year' in nightlight_df.columns:
    nightlight_df = nightlight_df[
        (nightlight_df['Year'] >= 2014) & (nightlight_df['Year'] <= 2024)
    ].copy()
    print(f"\nFiltered to 2014-2024: {len(nightlight_df)} observations")

# ============================================================================
# STEP 7: VALIDATE NIGHTLIGHT DATASET
# ============================================================================
print("\n" + "="*70)
print("STEP 7: VALIDATING NIGHTLIGHT DATASET")
print("="*70)

# 7.1 Check for missing values
print("\n7.1 Checking for missing values:")
print("-" * 50)
missing_nl = nightlight_df.isnull().sum()
print(missing_nl[missing_nl > 0] if missing_nl.sum() > 0 else "✓ No missing values found")

# 7.2 Check for duplicates
print("\n7.2 Checking for duplicate rows:")
print("-" * 50)
duplicates_nl = nightlight_df.duplicated().sum()
print(f"Number of duplicate rows: {duplicates_nl}")

# 7.3 Validate data types
print("\n7.3 Validating data types:")
print("-" * 50)
print(nightlight_df.dtypes)

# 7.4 Check nightlight values
print("\n7.4 Checking nightlight values:")
print("-" * 50)
if 'NightLight' in nightlight_df.columns:
    print(f"Min nightlight: {nightlight_df['NightLight'].min():.2f}")
    print(f"Max nightlight: {nightlight_df['NightLight'].max():.2f}")
    print(f"Mean nightlight: {nightlight_df['NightLight'].mean():.2f}")

# 7.5 Validate Year and Month columns
print("\n7.5 Validating Year and Month columns:")
print("-" * 50)
if 'Year' in nightlight_df.columns and 'Month' in nightlight_df.columns:
    print(f"Year range: {nightlight_df['Year'].min()} to {nightlight_df['Year'].max()}")
    print(f"Month range: {nightlight_df['Month'].min()} to {nightlight_df['Month'].max()}")
    print(f"Unique Year-Month combinations: {len(nightlight_df.groupby(['Year', 'Month']))}")
else:
    raise ValueError("'Year' and/or 'Month' columns not found in nightlight dataset")

# ============================================================================
# STEP 8: MERGE TEMPERATURE AND NIGHTLIGHT DATASETS
# ============================================================================
print("\n" + "="*70)
print("STEP 8: MERGING DATASETS")
print("="*70)

# Select only necessary columns
temp_clean = monthly_temp_filtered[['Year', 'Month', 'Avg_Monthly_Temperature']].copy()
nightlight_clean = nightlight_df[['Year', 'Month', 'NightLight']].copy()

# Merge datasets on Year and Month
merged_df = pd.merge(
    temp_clean,
    nightlight_clean,
    on=['Year', 'Month'],
    how='inner'  # Only keep months present in both datasets
)

# Sort by Year and Month for chronological order
merged_df = merged_df.sort_values(['Year', 'Month']).reset_index(drop=True)

print("\nMerged dataset:")
print("-" * 50)
print(merged_df.head(10))
print(f"\nFinal dataset shape: {merged_df.shape}")
print(f"Final columns: {merged_df.columns.tolist()}")
print(f"Date range: {merged_df['Year'].min()}-{merged_df['Month'].min()} to {merged_df['Year'].max()}-{merged_df['Month'].max()}")

# ============================================================================
# STEP 9: FINAL VALIDATION OF MERGED DATASET
# ============================================================================
print("\n" + "="*70)
print("STEP 9: FINAL VALIDATION OF MERGED DATASET")
print("="*70)

# Check for missing values
print("\n9.1 Final missing value check:")
print("-" * 50)
missing_final = merged_df.isnull().sum()
print(missing_final[missing_final > 0] if missing_final.sum() > 0 else "✓ No missing values in final dataset")

# Check for duplicates
print("\n9.2 Duplicate check:")
print("-" * 50)
duplicates_final = merged_df.duplicated(subset=['Year', 'Month']).sum()
print(f"Duplicate Year-Month combinations: {duplicates_final}")

# Check data ranges
print("\n9.3 Data range validation:")
print("-" * 50)
print(f"Year range: {merged_df['Year'].min()} - {merged_df['Year'].max()}")
print(f"Month range: {merged_df['Month'].min()} - {merged_df['Month'].max()}")
print(f"Nightlight range: {merged_df['NightLight'].min():.2f} - {merged_df['NightLight'].max():.2f}")
print(f"Temperature range: {merged_df['Avg_Monthly_Temperature'].min():.2f}°C - {merged_df['Avg_Monthly_Temperature'].max():.2f}°C")

# Validate data types
print("\n9.4 Data type validation:")
print("-" * 50)
print(merged_df.dtypes)

# ============================================================================
# STEP 10: EXPORT CLEANED DATASET
# ============================================================================
print("\n" + "="*70)
print("STEP 10: EXPORTING CLEANED DATASET")
print("="*70)

output_file = 'outputs/csv/Chennai_Monthly_Aggregated.csv'
merged_df.to_csv(output_file, index=False)
print(f"\n✓ Cleaned dataset exported to: {output_file}")
print(f"✓ Dataset ready for exploratory data analysis and modeling")
print(f"✓ Total observations: {len(merged_df)} monthly records")

print("\n" + "="*70)
print("DATA PREPROCESSING COMPLETE")
print("="*70)
print("\nKey Benefits of Monthly Resolution:")
print(f"  • More data points: {len(merged_df)}")