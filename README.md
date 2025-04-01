# Fake Medicine Detection System

A machine learning-based system for detecting fake medicines through image analysis.

## Overview

This application uses machine learning techniques to analyze medicine images and determine whether they are genuine or fake. 
It provides a user-friendly Streamlit interface for uploading images, training models, and visualizing results.

## Features

- **Image Analysis**: Extract key features from medicine images
- **Machine Learning**: Train models to classify medicines as genuine or fake
- **Visualization**: Visualize model performance and prediction results
- **User Interface**: Easy-to-use Streamlit interface for interacting with the system

## Requirements

- Python 3.7+
- Streamlit
- scikit-learn
- numpy
- pandas
- matplotlib
- seaborn
- OpenCV
- Pillow
- skimage

## How to Use

1. Run the application:
   ```
   streamlit run app.py
   ```

2. Use the navigation in the sidebar to:
   - Train a new model
   - Detect fake medicines from uploaded images
   - Evaluate model performance

## Model Training

The system allows training different types of machine learning models:
- Random Forest (default)
- Support Vector Machine (SVM)
- XGBoost

Feature extraction methods:
- HOG (Histogram of Oriented Gradients)
- Color Histograms
- Combined (both HOG and Color Histograms)

## Detection

Upload an image of a medicine or use one of the provided sample images to get a prediction on whether it's genuine or fake.

## Important Notes

This is a demonstration system and should not be used as the sole method for determining the authenticity of medicines. If you suspect a medicine is counterfeit, please consult with a healthcare professional or regulatory authority.

## License

MIT License
