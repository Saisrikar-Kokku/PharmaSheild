import streamlit as st
import os
import numpy as np
import pandas as pd
from PIL import Image
import io
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

from model import train_model, load_model, predict_image
from preprocessing import preprocess_image, extract_features
from utils import display_metrics, plot_feature_importance
from sample_data import load_sample_data, get_sample_images

# Set page configuration
st.set_page_config(
    page_title="Fake Medicine Detection System",
    page_icon="💊",
    layout="wide",
)

# Define the main title and description
st.title("Fake Medicine Detection System")
st.markdown("""
This application uses machine learning to detect fake medicines based on image analysis.
Upload an image of a medicine or use one of our sample images to get started.
""")

# Sidebar for navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Choose the app mode", ["Home", "Train Model", "Detect Fake Medicine", "Model Evaluation"])

# Initialize session state for model
if 'model' not in st.session_state:
    st.session_state.model = None
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'features' not in st.session_state:
    st.session_state.features = None
if 'labels' not in st.session_state:
    st.session_state.labels = None
if 'evaluation_metrics' not in st.session_state:
    st.session_state.evaluation_metrics = None

# Home page
if app_mode == "Home":
    st.header("About Fake Medicine Detection")
    
    st.markdown("""
    ### What are Fake Medicines?
    Counterfeit medicines are fake medicines that are designed to mimic real medicines.
    They may contain harmful ingredients, incorrect doses, or no active ingredients at all.
    
    ### How Does This System Work?
    1. **Image Analysis**: The system analyzes various visual features of medicine images
    2. **Feature Extraction**: Key characteristics are extracted from the images
    3. **Machine Learning**: A trained model classifies the medicine as real or fake
    4. **Results**: Prediction results and confidence scores are displayed
    
    ### Getting Started:
    - Use the sidebar to navigate to different sections
    - Train a new model or use the pre-trained model
    - Upload your own medicine images for detection
    - Explore model evaluation metrics
    """)
    
    st.info("Navigate to 'Train Model' to train a new model or 'Detect Fake Medicine' to test an image.")

# Train Model page
elif app_mode == "Train Model":
    st.header("Train Model")
    
    # Model training options
    st.subheader("Model Configuration")
    
    col1, col2 = st.columns(2)
    with col1:
        model_type = st.selectbox("Select Model Type", ["Random Forest", "SVM", "XGBoost"])
        test_size = st.slider("Test Size", 0.1, 0.5, 0.2, 0.05)
        
    with col2:
        feature_extraction = st.selectbox("Feature Extraction Method", ["HOG", "Color Histograms", "Combined"])
        random_state = st.number_input("Random State", 0, 100, 42, 1)
    
    # Train button
    if st.button("Train Model"):
        with st.spinner("Loading sample data..."):
            X, y = load_sample_data(feature_extraction_method=feature_extraction)
            
        if X is not None and y is not None:
            with st.spinner("Training model..."):
                model, evaluation_metrics, feature_names = train_model(
                    X, y, 
                    model_type=model_type,
                    test_size=test_size,
                    random_state=random_state
                )
                
                # Save to session state
                st.session_state.model = model
                st.session_state.model_trained = True
                st.session_state.features = feature_names
                st.session_state.evaluation_metrics = evaluation_metrics
                
                st.success("Model trained successfully!")
                
                # Display metrics
                st.subheader("Training Results")
                display_metrics(evaluation_metrics)
                
                # Plot feature importance if available
                if model_type in ["Random Forest", "XGBoost"]:
                    st.subheader("Feature Importance")
                    fig = plot_feature_importance(model, feature_names)
                    st.pyplot(fig)
        else:
            st.error("Could not load sample data. Please try again.")

# Detect Fake Medicine page
elif app_mode == "Detect Fake Medicine":
    st.header("Detect Fake Medicine")
    
    # Check if model is trained or load default
    if not st.session_state.model_trained:
        st.info("No model has been explicitly trained. Using default pre-trained model.")
        # Load default model
        st.session_state.model = load_model()
        st.session_state.model_trained = True
    
    # Image upload option
    st.subheader("Upload Medicine Image")
    
    input_option = st.radio("Select input option:", ["Upload your own image", "Use sample image"])
    
    image = None
    
    if input_option == "Upload your own image":
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
    else:  # Use sample image
        sample_images = get_sample_images()
        if sample_images:
            sample_choice = st.selectbox("Select a sample image:", list(sample_images.keys()))
            image = sample_images[sample_choice]
            st.image(image, caption=f"Sample: {sample_choice}", use_container_width=True)
        else:
            st.error("No sample images available")
    
    # Perform detection
    if image is not None and st.button("Detect"):
        with st.spinner("Analyzing image..."):
            # Preprocess image and extract features
            processed_image = preprocess_image(image)
            features = extract_features(processed_image, method="Combined")
            
            # Make prediction
            prediction, probability = predict_image(st.session_state.model, features)
            
            # Display results
            st.subheader("Detection Results")
            
            result_col1, result_col2 = st.columns(2)
            
            with result_col1:
                if prediction[0] == 1:
                    st.error(f"⚠️ **FAKE MEDICINE DETECTED** ⚠️")
                else:
                    st.success(f"✅ **GENUINE MEDICINE** ✅")
                    
            with result_col2:
                confidence = float(probability[0][int(prediction[0])]) * 100
                st.metric("Confidence", f"{confidence:.2f}%")
            
            st.write(f"**Detailed Analysis:** The model is {confidence:.2f}% confident in this prediction.")
            
            # Warning/notice
            if prediction[0] == 1:
                st.warning("""
                **Important Notice:** This is only a machine learning prediction and should not be taken as definitive proof. 
                If you suspect a counterfeit medicine, please consult with a healthcare professional or regulatory authority.
                """)

# Model Evaluation page
elif app_mode == "Model Evaluation":
    st.header("Model Evaluation")
    
    if st.session_state.model_trained and st.session_state.evaluation_metrics is not None:
        metrics = st.session_state.evaluation_metrics
        
        st.subheader("Performance Metrics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Accuracy", f"{metrics['accuracy']:.4f}")
        with col2:
            st.metric("Precision", f"{metrics['precision']:.4f}")
        with col3:
            st.metric("Recall", f"{metrics['recall']:.4f}")
            
        st.metric("F1 Score", f"{metrics['f1']:.4f}")
        
        # Confusion Matrix
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
        ax.set_title('Confusion Matrix')
        ax.set_xticklabels(['Genuine', 'Fake'])
        ax.set_yticklabels(['Genuine', 'Fake'])
        st.pyplot(fig)
        
        # ROC Curve
        st.subheader("ROC Curve")
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(metrics['fpr'], metrics['tpr'], label=f'AUC = {metrics["auc"]:.4f}')
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve')
        ax.legend(loc='lower right')
        st.pyplot(fig)
        
        # Classification Report
        st.subheader("Classification Report")
        st.text(metrics['classification_report'])
        
    else:
        st.info("No model has been trained yet. Please go to the 'Train Model' page to train a model first.")
        
        # Show example evaluation
        st.subheader("Example Evaluation Metrics")
        st.write("Here's an example of what the evaluation metrics will look like once you train a model:")
        
        # Create dummy metrics for display
        accuracy = 0.92
        precision = 0.94
        recall = 0.91
        f1 = 0.92
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Accuracy", f"{accuracy:.4f}")
        with col2:
            st.metric("Precision", f"{precision:.4f}")
        with col3:
            st.metric("Recall", f"{recall:.4f}")
            
        st.metric("F1 Score", f"{f1:.4f}")
        
        st.write("Note: These are example values, not actual model performance.")
