# Heart Disease Prediction Application

## Overview
This repository contains a Streamlit-based web application for predicting the likelihood of heart disease based on various health metrics. The application uses a K-Nearest Neighbors (KNN) classifier trained on a dataset of heart disease indicators.

## Contents

1. **Dataset**: `dataset.csv`
   - A CSV file containing heart disease-related data for training the model.

2. **Main Application**: `streamlit_app.py`
   - The core Python script that runs the Streamlit application.
   - Implements data preprocessing, model training, and user interface.

3. **Requirements**: `requirements.txt`
   - Lists the Python packages required to run the application.

4. **Streamlit Configuration**: `.streamlit/config.toml`
   - Contains custom theming for the Streamlit application.

5. **License**: `LICENSE`
   - MIT License file detailing the terms of use for this project.

6. **README**: `README.md`
   - Provides an overview of the application, its features, and usage instructions.

## Features

- Interactive user input for various health metrics
- Real-time prediction of heart disease likelihood
- Custom-themed Streamlit interface

## How It Works

1. The application loads and preprocesses the heart disease dataset.
2. A KNN classifier is trained on the preprocessed data.
3. Users input their health metrics through an interactive form.
4. The model predicts the likelihood of heart disease based on the input.
5. Results are displayed with a cautionary note about consulting healthcare professionals.

## Input Parameters

- Age
- Sex
- Chest Pain Type
- Resting Blood Pressure
- Serum Cholesterol
- Fasting Blood Sugar
- Resting Electrocardiographic Results
- Maximum Heart Rate Achieved
- Exercise-Induced Angina
- ST Depression
- Slope of Peak Exercise ST Segment
- Number of Major Vessels Colored by Fluoroscopy
- Thalassemia

## Output

The application provides a binary prediction:
- Low Probability of Heart Disease
- High Probability of Heart Disease

## Important Note

The application includes a disclaimer emphasizing that results should be used for reference only and users should consult authorized health professionals for medical advice.

## Setup and Usage

1. Install required packages: `pip install -r requirements.txt`
2. Run the Streamlit app: `streamlit run streamlit_app.py`
3. Access the application through a web browser

## Customization

The application's appearance can be customized by modifying the `.streamlit/config.toml` file.
