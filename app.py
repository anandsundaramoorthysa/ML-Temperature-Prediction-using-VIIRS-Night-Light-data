"""
PHASE 6: FLASK WEB APPLICATION
================================

This Flask application provides an interactive web interface for:
1. Project overview and methodology
2. Dataset information
3. Model comparison and results
4. Temperature prediction using the best model

Academic Rationale:
- Web interface makes research accessible to non-technical audiences
- Interactive predictions demonstrate model utility
- Visualization of results enhances communication
- Enables real-time exploration of urbanization-temperature relationships

Backend-Frontend Interaction:
- Flask backend handles model loading and predictions
- HTML/CSS/JavaScript frontend provides user interface
- JSON API endpoints allow programmatic access
- Server-side model execution ensures security and consistency
"""

from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

# ============================================================================
# LOAD MODELS AND DATA
# ============================================================================
print("Loading models...")

# Load best model (we'll determine this from saved info)
try:
    with open('outputs/csv/best_model_info.csv', 'r') as f:
        best_model_df = pd.read_csv(f)
        best_model_name = best_model_df['Best_Model'].values[0]
except:
    # Default to gradient boosting if file doesn't exist
    best_model_name = "Gradient Boosting"

# Load all models
models = {}
model_files = {
    'Linear Regression': 'models/model_linear_regression.pkl',
    'Random Forest': 'models/model_random_forest.pkl',
    'Gradient Boosting': 'models/model_gradient_boosting.pkl'
}

for model_name, model_file in model_files.items():
    try:
        with open(model_file, 'rb') as f:
            models[model_name] = pickle.load(f)
        print(f"✓ Loaded {model_name}")
    except FileNotFoundError:
        print(f"⚠ Warning: {model_file} not found")

# Load model comparison results
try:
    model_comparison = pd.read_csv('outputs/csv/model_comparison_results.csv')
except:
    model_comparison = None

# Load correlation results
try:
    correlation_results = pd.read_csv('outputs/csv/correlation_analysis_results.csv')
except:
    correlation_results = None

# Load feature importance
try:
    feature_importance = pd.read_csv('outputs/csv/feature_importance_comparison.csv')
except:
    feature_importance = None

print("Models loaded successfully!")

# ============================================================================
# ROUTES
# ============================================================================

@app.route('/', methods=['GET', 'POST'])
def index():
    """
    Home page - Temperature prediction tool
    """
    if request.method == 'POST':
        try:
            # Get input from form
            year = float(request.form.get('year'))
            month = float(request.form.get('month'))
            nightlight = float(request.form.get('nightlight'))
            
            # Validate inputs
            if year < 2014:
                return render_template('index.html',
                                     error="Year must be 2014 or later")
            
            # Check if prediction is extrapolation (beyond training data)
            is_extrapolation = year > 2024
            
            if month < 1 or month > 12:
                return render_template('index.html',
                                     error="Month must be between 1 and 12")
            
            if nightlight < 0 or nightlight > 100:
                return render_template('index.html',
                                     error="Nightlight intensity should be between 0 and 100")
            
            # Prepare input for prediction (Year, Month, NightLight)
            input_data = np.array([[year, month, nightlight]])
            
            # Get predictions from all available models
            predictions = {}
            for model_name, model in models.items():
                try:
                    pred = model.predict(input_data)[0]
                    predictions[model_name] = round(pred, 3)
                except:
                    predictions[model_name] = "Error"
            
            # Get best model prediction
            best_prediction = predictions.get(best_model_name, "N/A")
            
            return render_template('index.html',
                                 year=year,
                                 month=month,
                                 nightlight=nightlight,
                                 predictions=predictions,
                                 best_model=best_model_name,
                                 best_prediction=best_prediction,
                                 is_extrapolation=is_extrapolation)
        
        except Exception as e:
            return render_template('index.html',
                                 error=f"Error: {str(e)}")
    
    # GET request - show form
    return render_template('index.html')

@app.route('/analysis', methods=['GET', 'POST'])
def research():
    """
    Research page - Dataset, models, and analysis information
    """
    # Load dataset statistics
    try:
        sample_data = pd.read_csv('outputs/csv/Chennai_Monthly_Aggregated.csv')
        data_stats = {
            'total_samples': len(sample_data),
            'year_range': f"{int(sample_data['Year'].min())} - {int(sample_data['Year'].max())}",
            'nightlight_range': f"{sample_data['NightLight'].min():.2f} - {sample_data['NightLight'].max():.2f}",
            'temp_range': f"{sample_data['Avg_Monthly_Temperature'].min():.2f}°C - {sample_data['Avg_Monthly_Temperature'].max():.2f}°C"
        }
    except:
        data_stats = None
    
    predictions = None
    prediction_input = None
    is_extrapolation = False
    
    if request.method == 'POST':
        try:
            # Get input from form
            year = float(request.form.get('year'))
            month = float(request.form.get('month'))
            nightlight = float(request.form.get('nightlight'))
            
            # Validate inputs
            if year < 2014:
                return render_template('analysis.html',
                                     error="Year must be 2014 or later",
                                     data_stats=data_stats,
                                     model_comparison=model_comparison.to_dict('records') if model_comparison is not None else None,
                                     best_model=best_model_name,
                                     feature_importance=feature_importance.to_dict('records') if feature_importance is not None else None,
                                     correlation_results=correlation_results.to_dict('records') if correlation_results is not None else None)
            
            if month < 1 or month > 12:
                return render_template('analysis.html',
                                     error="Month must be between 1 and 12",
                                     data_stats=data_stats,
                                     model_comparison=model_comparison.to_dict('records') if model_comparison is not None else None,
                                     best_model=best_model_name,
                                     feature_importance=feature_importance.to_dict('records') if feature_importance is not None else None,
                                     correlation_results=correlation_results.to_dict('records') if correlation_results is not None else None)
            
            if nightlight < 0 or nightlight > 100:
                return render_template('analysis.html',
                                     error="Nightlight intensity should be between 0 and 100",
                                     data_stats=data_stats,
                                     model_comparison=model_comparison.to_dict('records') if model_comparison is not None else None,
                                     best_model=best_model_name,
                                     feature_importance=feature_importance.to_dict('records') if feature_importance is not None else None,
                                     correlation_results=correlation_results.to_dict('records') if correlation_results is not None else None)
            
            # Check if prediction is extrapolation
            is_extrapolation = year > 2024
            
            # Prepare input for prediction (Year, Month, NightLight)
            input_data = np.array([[year, month, nightlight]])
            
            # Get predictions from all available models
            predictions = {}
            for model_name, model in models.items():
                try:
                    pred = model.predict(input_data)[0]
                    predictions[model_name] = round(pred, 3)
                except:
                    predictions[model_name] = "Error"
            
            prediction_input = {
                'year': year,
                'month': month,
                'nightlight': nightlight
            }
        
        except Exception as e:
            return render_template('analysis.html',
                                 error=f"Error: {str(e)}",
                                 data_stats=data_stats,
                                 model_comparison=model_comparison.to_dict('records') if model_comparison is not None else None,
                                 best_model=best_model_name,
                                 feature_importance=feature_importance.to_dict('records') if feature_importance is not None else None,
                                 correlation_results=correlation_results.to_dict('records') if correlation_results is not None else None)
    
    error_message = None
    if request.method == 'GET':
        error_message = request.args.get('error')
    
    return render_template('analysis.html',
                         data_stats=data_stats,
                         model_comparison=model_comparison.to_dict('records') if model_comparison is not None else None,
                         best_model=best_model_name,
                         feature_importance=feature_importance.to_dict('records') if feature_importance is not None else None,
                         correlation_results=correlation_results.to_dict('records') if correlation_results is not None else None,
                         predictions=predictions,
                         prediction_input=prediction_input,
                         is_extrapolation=is_extrapolation,
                         error=error_message)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """
    API endpoint for programmatic predictions
    """
    try:
        data = request.get_json()
        year = float(data.get('year'))
        month = float(data.get('month'))
        nightlight = float(data.get('nightlight'))
        
        # Validate inputs
        if year < 2014:
            return jsonify({'success': False, 'error': 'Year must be 2014 or later'}), 400
        
        # Check if prediction is extrapolation
        is_extrapolation = year > 2024
        if month < 1 or month > 12:
            return jsonify({'success': False, 'error': 'Month must be between 1 and 12'}), 400
        if nightlight < 0 or nightlight > 100:
            return jsonify({'success': False, 'error': 'Nightlight must be between 0 and 100'}), 400
        
        # Prepare input (Year, Month, NightLight)
        input_data = np.array([[year, month, nightlight]])
        
        # Get predictions
        predictions = {}
        for model_name, model in models.items():
            try:
                pred = model.predict(input_data)[0]
                predictions[model_name] = round(pred, 3)
            except:
                predictions[model_name] = None
        
        return jsonify({
            'success': True,
            'input': {'year': year, 'month': month, 'nightlight': nightlight},
            'predictions': predictions,
            'best_model': best_model_name,
            'is_extrapolation': is_extrapolation,
            'warning': 'This is an extrapolation beyond training data (2014-2024). Predictions may be less reliable.' if is_extrapolation else None
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

# ============================================================================
# RUN APPLICATION
# ============================================================================

if __name__ == '__main__':
    # Hugging Face Spaces provides PORT environment variable (usually 7860)
    # For local development, default to 5000
    port = int(os.environ.get('PORT', 5000))
    # Run on all interfaces
    # For production on Hugging Face Spaces, debug should be False
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    app.run(debug=debug_mode, host='0.0.0.0', port=port)

