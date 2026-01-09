FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Create required directories
RUN mkdir -p models outputs/csv outputs/png static templates

# RUN ML PIPELINE (BUILD TIME)
RUN python 01_data_preprocessing.py && \
    python 02_exploratory_data_analysis.py && \
    python 03_ml_modeling.py && \
    python 04_model_evaluation.py && \
    python 05_feature_importance_interpretation.py

# Expose Hugging Face port
EXPOSE 7860

# Environment variables
ENV FLASK_APP=app.py
ENV FLASK_ENV=production
ENV PORT=7860

# Run Flask application
CMD ["python", "app.py"]
