# 🌡️ ML Temperature Prediction using VIIRS Night Light Data

---

## 📑 Table of Contents

* About Project
* Installation
* Run the Project
* Features
* Make a Prediction
* Contribution
* License
* Contact Me

---

## About Project

This project is a **Machine Learning–based Temperature Prediction System** developed for **Chennai, India**, using **VIIRS Nighttime Light intensity** as an input feature along with **Year** and **Month**.

The application predicts **monthly average temperature** and provides an interactive web interface to test predictions using trained machine learning models. The project demonstrates how satellite-derived nightlight data can be integrated with climate data for predictive analytics.

This is a **standalone ML project** focused on **prediction and evaluation**, not causal climate analysis.

---

## Installation

Follow these steps to set up the project locally:

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/anandsundaramoorthysa/ML-Temperature-Prediction-using-VIIRS-Night-Light-data.git
cd ML-Temperature-Prediction-using-VIIRS-Night-Light-data
```

### 2️⃣ Create a Virtual Environment (Optional but Recommended)

```bash
python -m venv venv
source venv/bin/activate        # On Linux/Mac
venv\Scripts\activate           # On Windows
```

### 3️⃣ Install Required Dependencies

```bash
pip install -r requirements.txt
```

---

## Run the Project

### 1️⃣ Run the Python Files

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

### 2️⃣ Run the Flask Application

```bash
python app.py
```

### 3️⃣ Open in Browser

Navigate to:

```
http://127.0.0.1:7860
```

## Make a Prediction

* Enter **Year** (2014 onwards)
* Enter **Month** (1–12)
* Enter **VIIRS Night Light Intensity**
* Click **Predict Temperature**
* View the predicted monthly average temperature

---

## Features

* 🌡️ **Monthly Temperature Prediction**
* 📊 **Multiple Model Comparison**

  * Multiple Linear Regression
  * Random Forest Regressor
  * Gradient Boosting Regressor
* 📈 **Model Evaluation Metrics**

  * RMSE
  * MAE
  * R² Score
* 🔍 **Feature Importance Analysis**
* 🖥️ **Interactive Web UI using Flask**
* 🚀 **Deployment-ready (Hugging Face / Render)**

---

## Contribution

Contributions are welcome!

If you would like to improve this project:

1. Fork the repository
2. Create a new feature branch
3. Commit your changes
4. Submit a pull request

Suggestions for improvements include:

* UI enhancements
* Model optimization
* Additional visualization features

---

## License

This project is licensed under the **MIT License**.
You are free to use, modify, and distribute this project with attribution.

---

## Contact Me

If you have any questions, feedback, or suggestions, feel free to reach out to the author:

**ANAND SUNDARAMOORTHY SA**
📧 [sanand03072005@gmail.com](mailto:sanand03072005@gmail.com?subject=Question%20about%20ML%20Temperature%20Prediction%20Project&body=Dear%20Author%2C%0A%0AI%20have%20a%20question%20regarding%20the%20ML%20Temperature%20Prediction%20project%2E%0A%0A%5BYour%20Question%20Here%5D%0A%0AThank%20you%21%0A%5BYour%20Name%5D)
