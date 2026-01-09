# Quick Start Guide

## How to Run This Project

Follow these steps to run the complete analysis and web application with **monthly resolution** data.

---

## Step 1: Install Dependencies

### Option A: Using Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# Install required packages
pip install -r requirements.txt
```

### Option B: Direct Installation

```bash
pip install -r requirements.txt
```

---

## Step 2: Run Analysis Scripts (In Order)

**Important:** Run these scripts in numerical order (01 → 05). Each script depends on outputs from previous ones.

```bash
# Step 1: Preprocess and validate data (Monthly aggregation, 2014-2024)
python 01_data_preprocessing.py

# Step 2: Exploratory data analysis (Monthly EDA)
python 02_exploratory_data_analysis.py

# Step 3: Train machine learning models (Time-based split, 3 features)
python 03_ml_modeling.py

# Step 4: Evaluate and compare models (Overfitting detection)
python 04_model_evaluation.py

# Step 5: Feature importance analysis (Year, Month, NightLight)
python 05_feature_importance_interpretation.py
```

**Expected outputs after running all scripts:**

All outputs are organized in folders:
- `outputs/csv/` - All CSV files:
  - `Chennai_Monthly_Aggregated.csv` - Processed monthly dataset (132 observations)
  - `model_comparison_results.csv` - Model performance metrics
  - `best_model_info.csv` - Best model selection
  - `feature_importance_comparison.csv` - Feature importance analysis
  - `correlation_analysis_results.csv` - Correlation coefficients
- `outputs/png/` - All visualization PNG files (8 files):
  - Monthly temperature and nightlight trends
  - Seasonal patterns
  - Scatter plots
  - Correlation heatmaps
  - Model comparisons
- `models/` - Trained model files (.pkl) (3 files):
  - `model_linear_regression.pkl`
  - `model_random_forest.pkl`
  - `model_gradient_boosting.pkl`

---

## Step 3: Start the Web Application

```bash
python app.py
```

**Expected output:**
```
Loading models...
✓ Loaded Linear Regression
✓ Loaded Random Forest
✓ Loaded Gradient Boosting
Models loaded successfully!
 * Running on http://127.0.0.1:5000
```

---

## Step 4: Access the Web Application

Open your web browser and go to:
```
http://localhost:5000
```

or

```
http://127.0.0.1:5000
```

---

## What You'll See

### Web Application Pages:

1. **Home** (`/`) - Project overview and key findings (monthly analysis)
2. **Dataset** (`/dataset`) - Data information and statistics (monthly data, 2014-2024)
3. **Models** (`/models`) - Model comparison and performance
4. **Predict** (`/predict`) - Interactive monthly temperature prediction tool
5. **About** (`/about`) - Methodology and limitations

### Try the Prediction Tool:

1. Go to the **Predict** page
2. Enter:
   - **Year:** e.g., 2024 (must be 2014-2024)
   - **Month:** e.g., 6 (June) - select from dropdown
   - **Nightlight:** e.g., 22.0 (typically ranges ~10-25)
3. Click **"Predict Temperature"**
4. See predictions from all models for that specific month

---

## Troubleshooting

### Problem: "Module not found" error

**Solution:**
```bash
pip install -r requirements.txt
```

### Problem: Model files (.pkl) not found

**Solution:**
Make sure you ran all 5 analysis scripts first:
```bash
python 01_data_preprocessing.py
python 02_exploratory_data_analysis.py
python 03_ml_modeling.py
python 04_model_evaluation.py
python 05_feature_importance_interpretation.py
```

### Problem: Monthly nightlight file not found

**Solution:**
Ensure you have `Chennai_VIIRS_Monthly_Nightlights_2014_2024.csv` or `Chennai_VIIRS_Monthly_Nightlights_2012_2024.csv` in the project directory. The script will filter to 2014-2024 automatically.

### Problem: Port 5000 already in use

**Solution:**
Either:
- Stop the other application using port 5000
- Modify `app.py` to use a different port:
  ```python
  app.run(debug=True, host='0.0.0.0', port=5001)  # Change 5000 to 5001
  ```

### Problem: Data files not found

**Solution:**
Ensure these files are in the project directory:
- `Chennai.csv` - Temperature data
- `Chennai_VIIRS_Monthly_Nightlights_2014_2024.csv` (or similar) - Monthly nightlight data

---

## Key Differences from Yearly Analysis

### Monthly Resolution Benefits:

1. **More Data Points:**
   - **Monthly:** 132 observations (11 years × 12 months)
   - **Yearly:** 12 observations
   - **Improvement:** 11x more data reduces overfitting risk

2. **Better Temporal Resolution:**
   - Captures seasonal patterns
   - Reveals intra-annual variations
   - More robust model training

3. **Time-Based Validation:**
   - Earlier months → Training (~105 samples, 80%)
   - Recent months → Testing (~27 samples, 20%)
   - Preserves temporal order (prevents data leakage)

4. **Three Features:**
   - **Year:** Long-term trends
   - **Month:** Seasonal patterns
   - **NightLight:** Human activity proxy

---

## Quick Commands Summary

```bash
# Setup (one time)
python -m venv venv
venv\Scripts\activate  # Windows
# OR
source venv/bin/activate  # Mac/Linux
pip install -r requirements.txt

# Run analysis (in order) - Monthly Resolution
python 01_data_preprocessing.py      # Monthly aggregation, 2014-2024
python 02_exploratory_data_analysis.py
python 03_ml_modeling.py            # Time-based split, Year+Month+NightLight
python 04_model_evaluation.py       # Overfitting detection
python 05_feature_importance_interpretation.py

# Start web app
python app.py

# Open browser to http://localhost:5000
```

---

## Verification Checklist

After running all scripts, verify you have:

- [ ] `outputs/csv/Chennai_Monthly_Aggregated.csv` file created (132 rows)
- [ ] `models/model_linear_regression.pkl` exists
- [ ] `models/model_random_forest.pkl` exists
- [ ] `models/model_gradient_boosting.pkl` exists
- [ ] `outputs/csv/model_comparison_results.csv` exists
- [ ] `outputs/csv/best_model_info.csv` exists
- [ ] `outputs/csv/feature_importance_comparison.csv` exists (3 features: Year, Month, NightLight)
- [ ] Visualization PNG files created in `outputs/png/` (8 files)
- [ ] Web app starts without errors
- [ ] Can access http://localhost:5000 in browser
- [ ] Prediction form accepts Year (2014-2024), Month (1-12), Nightlight

---

## Expected Runtime

- **Data preprocessing:** ~1-2 minutes (monthly aggregation)
- **EDA:** ~30-60 seconds (generates 5 visualizations)
- **Model training:** ~10-30 seconds (3 models)
- **Model evaluation:** ~10 seconds
- **Feature importance:** ~10 seconds
- **Total analysis time:** ~3-5 minutes

---

## Important Notes

### Data Requirements:

- **Temperature data:** `Chennai.csv` (should contain `date` and `temperature_2m` columns)
- **Nightlight data:** `Chennai_VIIRS_Monthly_Nightlights_2014_2024.csv` (or similar with Year, Month, NightLight columns)

### Data Period:

- **Analysis period:** 2014-2024 only
- **Reason:** VIIRS monthly data availability limitation
- **Output:** 132 monthly observations (11 years × 12 months)

### Prediction Input:

- **Year:** Must be between 2014 and 2024
- **Month:** Must be between 1 (January) and 12 (December)
- **Nightlight:** Typically ranges from ~10 to ~25 based on historical data

---

## Need Help?

- Check `README.md` for detailed documentation
- Review individual script files for specific explanations
- See `DEPLOYMENT_GUIDE.md` for deployment instructions
- Check `UPDATED_PROJECT_SUMMARY.md` for summary of changes

---

**Ready to start? Run the commands above!** 🚀

**Remember:** This is monthly resolution analysis with improved statistical rigor compared to yearly analysis.
