import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import xgboost as xgb
import pickle
import os
import io
from PIL import Image

def train_model(X, y, model_type="Random Forest", test_size=0.2, random_state=42):
    """
    Train a machine learning model on the given data
    
    Args:
        X: Features
        y: Labels (0 for genuine, 1 for fake)
        model_type: Type of model to train (Random Forest, SVM, XGBoost)
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
        
    Returns:
        model: Trained model
        evaluation_metrics: Dictionary of evaluation metrics
        feature_names: List of feature names
    """
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Initialize model based on model_type with optimized hyperparameters
    if model_type == "Random Forest":
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=4,
            min_samples_leaf=2,
            max_features='sqrt',
            bootstrap=True,
            class_weight='balanced',
            random_state=random_state
        )
    elif model_type == "SVM":
        model = SVC(
            kernel='rbf',
            C=100,  # Increased regularization parameter
            gamma='auto',
            probability=True,
            class_weight='balanced',  # Handle imbalanced classes better
            random_state=random_state
        )
    elif model_type == "XGBoost":
        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=7,
            learning_rate=0.05,  # Lower learning rate for better generalization
            subsample=0.8,  # Use a subset of samples per tree
            colsample_bytree=0.8,  # Use a subset of features per tree
            min_child_weight=3,  # Control overfitting
            scale_pos_weight=1,  # Balance positive/negative weights
            random_state=random_state
        )
    else:
        # Default to Random Forest with optimized parameters
        model = RandomForestClassifier(
            n_estimators=200, 
            max_depth=15,
            class_weight='balanced',
            random_state=random_state
        )
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions on the test set
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    
    # Calculate evaluation metrics
    evaluation_metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='binary'),
        'recall': recall_score(y_test, y_pred, average='binary'),
        'f1': f1_score(y_test, y_pred, average='binary'),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred, target_names=['Genuine', 'Fake']),
    }
    
    # Calculate ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])
    evaluation_metrics['fpr'] = fpr
    evaluation_metrics['tpr'] = tpr
    evaluation_metrics['auc'] = auc(fpr, tpr)
    
    # Generate feature names (generic if not available)
    if hasattr(model, 'feature_names_in_'):
        feature_names = model.feature_names_in_
    else:
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    
    return model, evaluation_metrics, feature_names

def save_model(model, filename="fake_medicine_detection_model.pkl"):
    """
    Save the trained model to disk
    
    Args:
        model: Trained model to save
        filename: Filename to save the model
    """
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    return filename

def load_model(filename="fake_medicine_detection_model.pkl"):
    """
    Load a trained model from disk or create a default one if file doesn't exist
    
    Args:
        filename: Filename of the saved model
        
    Returns:
        Loaded model
    """
    # If model file exists, load it
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            model = pickle.load(f)
        return model
    
    # Otherwise, train a default model on sample data
    from sample_data import load_sample_data
    X, y = load_sample_data()
    model, _, _ = train_model(X, y)
    return model

def predict_image(model, features):
    """
    Predict whether an image contains fake medicine
    
    Args:
        model: Trained model
        features: Extracted features from the image
        
    Returns:
        prediction: 0 for genuine, 1 for fake
        probability: Probability scores for each class
    """
    # Reshape features if needed
    if features.ndim == 1:
        features = features.reshape(1, -1)
    
    # Make prediction
    prediction = model.predict(features)
    probability = model.predict_proba(features)
    
    return prediction, probability
