import numpy as np
import pandas as pd
from PIL import Image
import os
import io
import urllib.request
from io import BytesIO
import cv2
import random
from preprocessing import preprocess_image, extract_features

def generate_synthetic_data(n_samples=1000, feature_dim=100, class_balance=0.5, random_state=42):
    """
    Generate synthetic data for model training
    
    Args:
        n_samples: Number of samples to generate
        feature_dim: Dimensionality of feature vectors
        class_balance: Proportion of samples in the positive class (fake medicines)
        random_state: Random seed for reproducibility
        
    Returns:
        X: Feature matrix (n_samples x feature_dim)
        y: Labels (0 for genuine, 1 for fake)
    """
    np.random.seed(random_state)
    
    # Calculate number of samples in each class
    n_fake = int(n_samples * class_balance)
    n_genuine = n_samples - n_fake
    
    # Generate features for genuine medicines
    # Genuine medicines have more consistent features
    genuine_mean = np.random.rand(feature_dim) * 0.5
    genuine_std = np.random.rand(feature_dim) * 0.1
    X_genuine = np.random.normal(loc=genuine_mean, scale=genuine_std, size=(n_genuine, feature_dim))
    
    # Generate features for fake medicines
    # Fake medicines have more variable features
    fake_mean = np.random.rand(feature_dim) * 0.5 + 0.5
    fake_std = np.random.rand(feature_dim) * 0.3
    X_fake = np.random.normal(loc=fake_mean, scale=fake_std, size=(n_fake, feature_dim))
    
    # Combine features and labels
    X = np.vstack([X_genuine, X_fake])
    y = np.hstack([np.zeros(n_genuine), np.ones(n_fake)])
    
    # Shuffle data
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]
    
    return X, y

def create_synthetic_image(is_fake=False, size=(224, 224), random_state=None):
    """
    Create a synthetic medicine image for demonstration
    
    Args:
        is_fake: Whether to create a fake or genuine medicine image
        size: Size of the image
        random_state: Random seed
        
    Returns:
        image: PIL Image object
    """
    if random_state is not None:
        np.random.seed(random_state)
        random.seed(random_state)
    
    # Create blank image
    image = np.ones((size[0], size[1], 3), dtype=np.uint8) * 255
    
    # Draw pill or capsule shape
    pill_type = random.choice(['round', 'oval', 'capsule'])
    
    if pill_type == 'round':
        center = (size[0] // 2, size[1] // 2)
        radius = min(size) // 3
        
        # Main pill color
        if is_fake:
            # Fake pills may have inconsistent coloring or unusual colors
            color = (
                random.randint(100, 255),
                random.randint(100, 255),
                random.randint(100, 255)
            )
            # Add some imperfections
            noise = np.random.randint(0, 30, size=(size[0], size[1], 3), dtype=np.uint8)
            image = cv2.add(image, noise)
        else:
            # Genuine pills have more consistent, pharmaceutical colors
            color_choices = [
                (255, 255, 255),  # White
                (220, 220, 220),  # Light gray
                (200, 200, 250),  # Light blue
                (250, 200, 200),  # Light pink
                (200, 250, 200),  # Light green
                (250, 250, 200)   # Light yellow
            ]
            color = random.choice(color_choices)
        
        cv2.circle(image, center, radius, color, -1)
        
        # Add score line or imprint
        if not is_fake or random.random() < 0.7:  # Genuine pills always have clear markings
            line_color = (180, 180, 180)
            if random.random() < 0.5:
                # Score line
                cv2.line(
                    image, 
                    (center[0] - radius, center[1]), 
                    (center[0] + radius, center[1]), 
                    line_color, 
                    2
                )
            else:
                # Imprint/logo
                font = cv2.FONT_HERSHEY_SIMPLEX
                text = random.choice(['A', 'B', 'C', 'X', 'Y', 'Z', '1', '2', '3'])
                text_size = cv2.getTextSize(text, font, 1, 2)[0]
                text_x = center[0] - text_size[0] // 2
                text_y = center[1] + text_size[1] // 2
                cv2.putText(image, text, (text_x, text_y), font, 1, line_color, 2)
    
    elif pill_type == 'oval':
        center = (size[0] // 2, size[1] // 2)
        axes = (size[0] // 3, size[1] // 4)
        
        if is_fake:
            color = (
                random.randint(100, 255),
                random.randint(100, 255),
                random.randint(100, 255)
            )
            noise = np.random.randint(0, 30, size=(size[0], size[1], 3), dtype=np.uint8)
            image = cv2.add(image, noise)
        else:
            color_choices = [
                (255, 255, 255),  # White
                (220, 220, 220),  # Light gray
                (200, 200, 250),  # Light blue
                (250, 200, 200),  # Light pink
                (200, 250, 200),  # Light green
                (250, 250, 200)   # Light yellow
            ]
            color = random.choice(color_choices)
        
        cv2.ellipse(image, center, axes, 0, 0, 360, color, -1)
        
        # Add score line or imprint
        if not is_fake or random.random() < 0.7:
            line_color = (180, 180, 180)
            if random.random() < 0.5:
                # Score line
                cv2.line(
                    image, 
                    (center[0] - axes[0], center[1]), 
                    (center[0] + axes[0], center[1]), 
                    line_color, 
                    2
                )
            else:
                # Imprint/logo
                font = cv2.FONT_HERSHEY_SIMPLEX
                text = random.choice(['A', 'B', 'C', 'X', 'Y', 'Z', '1', '2', '3'])
                text_size = cv2.getTextSize(text, font, 1, 2)[0]
                text_x = center[0] - text_size[0] // 2
                text_y = center[1] + text_size[1] // 2
                cv2.putText(image, text, (text_x, text_y), font, 1, line_color, 2)
    
    elif pill_type == 'capsule':
        # Draw capsule
        center = (size[0] // 2, size[1] // 2)
        width = size[0] // 2
        height = size[1] // 4
        
        if is_fake:
            color1 = (
                random.randint(100, 255),
                random.randint(100, 255),
                random.randint(100, 255)
            )
            color2 = (
                random.randint(100, 255),
                random.randint(100, 255),
                random.randint(100, 255)
            )
            noise = np.random.randint(0, 30, size=(size[0], size[1], 3), dtype=np.uint8)
            image = cv2.add(image, noise)
        else:
            color_pairs = [
                ((255, 200, 200), (200, 200, 255)),  # Red + Blue
                ((255, 255, 200), (200, 255, 200)),  # Yellow + Green
                ((255, 200, 255), (200, 255, 255)),  # Purple + Cyan
                ((255, 200, 200), (255, 255, 200)),  # Red + Yellow
                ((200, 200, 255), (200, 255, 255))   # Blue + Cyan
            ]
            color1, color2 = random.choice(color_pairs)
        
        # Draw left half-circle
        cv2.circle(image, (center[0] - width//2, center[1]), height, color1, -1)
        # Draw right half-circle
        cv2.circle(image, (center[0] + width//2, center[1]), height, color2, -1)
        # Draw rectangle in the middle
        cv2.rectangle(
            image, 
            (center[0] - width//2, center[1] - height), 
            (center[0] + width//2, center[1] + height), 
            color1, 
            -1
        )
        # Draw right half of rectangle with second color
        cv2.rectangle(
            image, 
            (center[0], center[1] - height), 
            (center[0] + width//2, center[1] + height), 
            color2, 
            -1
        )
    
    # Apply slight blur to make it more realistic
    image = cv2.GaussianBlur(image, (5, 5), 0)
    
    # Add manufacturing details for genuine pills or poor imitations for fake
    if not is_fake:
        # Genuine pills have crisp, clear details
        font = cv2.FONT_HERSHEY_SIMPLEX
        batch_number = f"LOT {random.randint(10000, 99999)}"
        cv2.putText(image, batch_number, (10, size[1] - 20), font, 0.5, (100, 100, 100), 1)
    elif random.random() < 0.4:
        # Some fake pills try to imitate real ones but with errors
        font = cv2.FONT_HERSHEY_SIMPLEX
        # Intentionally misspelled or unusual batch number
        batch_number = f"L0T {random.randint(100, 999)}"
        cv2.putText(image, batch_number, (10, size[1] - 20), font, 0.5, (100, 100, 100), 1)
    
    # Convert to PIL Image
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    return image_pil

def load_sample_data(n_samples=200, feature_extraction_method="Combined", random_state=42):
    """
    Load or generate sample data for model training
    
    Args:
        n_samples: Number of samples to generate
        feature_extraction_method: Method to extract features from images
        random_state: Random seed for reproducibility
        
    Returns:
        X: Feature matrix
        y: Labels (0 for genuine, 1 for fake)
    """
    # Set random seed
    np.random.seed(random_state)
    random.seed(random_state)
    
    # Calculate samples per class (balanced)
    n_per_class = n_samples // 2
    
    features_list = []
    labels = []
    
    # Generate synthetic images and extract features
    for i in range(n_samples):
        is_fake = i >= n_per_class
        
        # Create synthetic image
        image = create_synthetic_image(
            is_fake=is_fake, 
            random_state=random_state + i
        )
        
        # Preprocess image
        processed_image = preprocess_image(image)
        
        # Extract features
        features = extract_features(processed_image, method=feature_extraction_method)
        
        features_list.append(features)
        labels.append(1 if is_fake else 0)
    
    # Convert to numpy arrays
    X = np.array(features_list)
    y = np.array(labels)
    
    return X, y

def get_sample_images(n_samples=5):
    """
    Get sample medicine images for demonstration
    
    Args:
        n_samples: Number of sample images to generate
        
    Returns:
        sample_images: Dictionary of sample images {name: PIL Image}
    """
    sample_images = {}
    
    # Generate genuine medicine samples
    for i in range(n_samples // 2):
        image = create_synthetic_image(is_fake=False, random_state=42 + i)
        sample_images[f"Genuine Medicine {i+1}"] = image
    
    # Generate fake medicine samples
    for i in range(n_samples - n_samples // 2):
        image = create_synthetic_image(is_fake=True, random_state=100 + i)
        sample_images[f"Fake Medicine {i+1}"] = image
    
    return sample_images
