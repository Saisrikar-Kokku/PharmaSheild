import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

def display_metrics(metrics):
    """
    Display evaluation metrics in a Streamlit app
    
    Args:
        metrics: Dictionary of evaluation metrics
    """
    # Create columns for metrics
    col1, col2, col3, col4 = st.columns(4)
    
    # Display metrics in each column
    with col1:
        st.metric("Accuracy", f"{metrics['accuracy']:.4f}")
    
    with col2:
        st.metric("Precision", f"{metrics['precision']:.4f}")
    
    with col3:
        st.metric("Recall", f"{metrics['recall']:.4f}")
    
    with col4:
        st.metric("F1 Score", f"{metrics['f1']:.4f}")

def plot_feature_importance(model, feature_names):
    """
    Plot feature importance for a trained model
    
    Args:
        model: Trained model with feature_importances_ attribute
        feature_names: List of feature names
        
    Returns:
        fig: Matplotlib figure
    """
    # Check if model has feature_importances_
    if not hasattr(model, 'feature_importances_'):
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Feature importance not available for this model type", 
                ha='center', va='center', transform=ax.transAxes)
        return fig
    
    # Get feature importances
    importances = model.feature_importances_
    
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]
    
    # Limit to top 20 features for better visualization
    top_k = min(20, len(indices))
    top_indices = indices[:top_k]
    
    # Rearrange feature names and importances
    top_feature_names = [feature_names[i] if i < len(feature_names) else f"feature_{i}" for i in top_indices]
    top_importances = importances[top_indices]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot importance bars
    bars = ax.barh(range(top_k), top_importances, align='center')
    
    # Set labels and title
    ax.set_yticks(range(top_k))
    ax.set_yticklabels(top_feature_names)
    ax.set_xlabel('Feature Importance')
    ax.set_title('Top Feature Importances')
    
    # Invert y-axis to have the most important feature at the top
    ax.invert_yaxis()
    
    return fig

def plot_confusion_matrix(y_true, y_pred, class_names=None):
    """
    Plot confusion matrix
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names of classes
        
    Returns:
        fig: Matplotlib figure
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    
    # Set labels and title
    if class_names is None:
        class_names = ['Genuine', 'Fake']
    
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    
    return fig

def plot_roc_curve(y_true, y_score):
    """
    Plot ROC curve
    
    Args:
        y_true: True labels
        y_score: Predicted probabilities for the positive class
        
    Returns:
        fig: Matplotlib figure
    """
    # Compute ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot ROC curve
    ax.plot(fpr, tpr, lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=2)
    
    # Set labels and title
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic')
    ax.legend(loc="lower right")
    
    return fig
