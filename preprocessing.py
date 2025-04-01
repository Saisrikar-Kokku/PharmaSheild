import numpy as np
import cv2
from PIL import Image
import io
import os
from skimage.feature import hog
from skimage import color, exposure

def preprocess_image(image, target_size=(224, 224)):
    """
    Preprocess an image for feature extraction with enhanced robustness
    
    Args:
        image: PIL Image or file path
        target_size: Size to resize the image to
        
    Returns:
        preprocessed_image: Numpy array of preprocessed image
    """
    # If image is a file path, load it
    if isinstance(image, str) and os.path.exists(image):
        image = Image.open(image)
    
    # If image is PIL Image, convert to numpy array
    if isinstance(image, Image.Image):
        # Ensure RGB format
        if image.mode != 'RGB':
            image = image.convert('RGB')
        # Resize the image
        image = image.resize(target_size)
        # Convert to numpy array
        img_array = np.array(image)
    else:
        # Assume it's already a numpy array
        img_array = image
        img_array = cv2.resize(img_array, target_size)
    
    # Convert to RGB if needed (OpenCV uses BGR)
    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    
    # Apply histogram equalization to improve contrast
    if len(img_array.shape) == 3:
        # Convert to LAB color space
        lab = cv2.cvtColor(img_array.astype(np.uint8), cv2.COLOR_RGB2LAB)
        # Split the LAB channels
        l, a, b = cv2.split(lab)
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        # Merge the enhanced L channel with A and B channels
        enhanced_lab = cv2.merge((cl, a, b))
        # Convert back to RGB
        img_array = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
    
    # Add Gaussian blur to reduce noise
    img_array = cv2.GaussianBlur(img_array, (3, 3), 0)
    
    # Normalize the image
    img_array = img_array.astype(np.float32) / 255.0
    
    return img_array

def extract_color_histogram(image, bins=32):
    """
    Extract color histograms from an image
    
    Args:
        image: Preprocessed image array
        bins: Number of bins for histogram
        
    Returns:
        features: Flattened histogram features
    """
    # Convert to float and normalize
    if image.max() > 1.0:
        image = image / 255.0
    
    # Extract histograms for each channel
    hist_r = np.histogram(image[:, :, 0], bins=bins, range=(0, 1))[0]
    hist_g = np.histogram(image[:, :, 1], bins=bins, range=(0, 1))[0]
    hist_b = np.histogram(image[:, :, 2], bins=bins, range=(0, 1))[0]
    
    # Normalize histograms
    hist_r = hist_r / np.sum(hist_r) if np.sum(hist_r) > 0 else hist_r
    hist_g = hist_g / np.sum(hist_g) if np.sum(hist_g) > 0 else hist_g
    hist_b = hist_b / np.sum(hist_b) if np.sum(hist_b) > 0 else hist_b
    
    # Concatenate histograms
    features = np.concatenate([hist_r, hist_g, hist_b])
    
    return features

def extract_hog_features(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2)):
    """
    Extract HOG (Histogram of Oriented Gradients) features from an image
    
    Args:
        image: Preprocessed image array
        orientations: Number of orientation bins
        pixels_per_cell: Size of a cell in pixels
        cells_per_block: Number of cells in each block
        
    Returns:
        features: HOG features
    """
    # Convert to grayscale
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = color.rgb2gray(image)
    else:
        gray = image
    
    # Extract HOG features
    features = hog(
        gray, 
        orientations=orientations, 
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block, 
        visualize=False,
        feature_vector=True
    )
    
    return features

def extract_texture_features(image):
    """
    Extract texture features using GLCM (Gray Level Co-occurrence Matrix)
    
    Args:
        image: Preprocessed image array
        
    Returns:
        features: Texture features
    """
    # Convert to grayscale and uint8
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = color.rgb2gray(image)
    else:
        gray = image
    
    gray = (gray * 255).astype(np.uint8)
    
    # Compute gradients (Sobel)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # Compute gradient magnitude and direction
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    
    # Normalize magnitude
    if magnitude.max() > 0:
        magnitude = magnitude / magnitude.max()
    
    # Compute histogram of gradient magnitudes
    hist_mag = np.histogram(magnitude, bins=32, range=(0, 1))[0]
    hist_mag = hist_mag / np.sum(hist_mag) if np.sum(hist_mag) > 0 else hist_mag
    
    return hist_mag

def extract_sift_features(image, n_features=100):
    """
    Extract SIFT (Scale-Invariant Feature Transform) features
    
    Args:
        image: Preprocessed image array
        n_features: Number of SIFT features to extract
        
    Returns:
        features: SIFT features
    """
    # Convert to grayscale and uint8
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = color.rgb2gray(image)
    else:
        gray = image
    
    gray = (gray * 255).astype(np.uint8)
    
    try:
        # Initialize SIFT detector
        sift = cv2.SIFT_create()
        
        # Detect keypoints and compute descriptors
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        
        # If no keypoints found, return zeros
        if descriptors is None or len(keypoints) == 0:
            return np.zeros(128)  # Standard SIFT dimension
        
        # Calculate average descriptor
        avg_descriptor = np.mean(descriptors, axis=0)
        
        # Ensure consistent length by returning the average descriptor
        return avg_descriptor
        
    except Exception as e:
        # In case of any errors, return a zero vector
        return np.zeros(128)  # Standard SIFT dimension

def extract_features(image, method="Combined"):
    """
    Extract features from an image using the specified method
    
    Args:
        image: Preprocessed image array
        method: Feature extraction method ("HOG", "Color Histograms", "Combined", "Advanced")
        
    Returns:
        features: Extracted features
    """
    if method == "HOG":
        features = extract_hog_features(image)
    elif method == "Color Histograms":
        features = extract_color_histogram(image)
    elif method == "Combined":
        hog_features = extract_hog_features(image)
        color_features = extract_color_histogram(image)
        texture_features = extract_texture_features(image)
        
        # Use SIFT features for more robust recognition
        try:
            sift_features = extract_sift_features(image)
            features = np.concatenate([hog_features, color_features, texture_features, sift_features])
        except:
            # If SIFT fails, fallback to original features
            features = np.concatenate([hog_features, color_features, texture_features])
    else:
        # Default to HOG
        features = extract_hog_features(image)
    
    return features
