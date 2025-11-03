import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd

# Try to import plotly, fall back to matplotlib if not available
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    st.sidebar.warning("Plotly not installed. Using matplotlib for charts.")

# Configure page
st.set_page_config(
    page_title="Brain Tumor Classification",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
IMG_SIZE = 128
INPUT_SHAPE = (128, 128, 3)
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .validation-pass {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #28a745;
        margin: 0.5rem 0;
        color: #155724;
    }
    .validation-fail {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #dc3545;
        margin: 0.5rem 0;
        color: #721c24;
    }
    .prediction-box {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #2E86AB;
        margin: 0.5rem 0;
    }
    .ensemble-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ================================
# CBAM Components
# ================================
@tf.keras.utils.register_keras_serializable()
class SpatialAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.conv = tf.keras.layers.Conv2D(1, 7, padding='same', activation='sigmoid', use_bias=False)

    def call(self, inputs):
        avg_out = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        max_out = tf.reduce_max(inputs, axis=-1, keepdims=True)
        concat = tf.concat([avg_out, max_out], axis=-1)
        return self.conv(concat) * inputs

    def get_config(self):
        return super().get_config()

@tf.keras.utils.register_keras_serializable()
class CBAM(tf.keras.layers.Layer):
    def __init__(self, ratio=8, **kwargs):
        super().__init__(**kwargs)
        self.ratio = ratio
        self.spatial_attention = SpatialAttention()

    def build(self, input_shape):
        ch = input_shape[-1]
        mid = max(1, ch // self.ratio)
        self.fc1 = tf.keras.layers.Dense(mid, activation='relu', kernel_initializer='he_normal', use_bias=True)
        self.fc2 = tf.keras.layers.Dense(ch, kernel_initializer='he_normal', use_bias=True)
        super().build(input_shape)

    def call(self, x):
        avg_pool = tf.reduce_mean(x, axis=[1,2], keepdims=True)
        max_pool = tf.reduce_max(x, axis=[1,2], keepdims=True)
        avg_out = self.fc2(self.fc1(avg_pool))
        max_out = self.fc2(self.fc1(max_pool))
        scale = tf.nn.sigmoid(avg_out + max_out)
        x = x * scale
        return self.spatial_attention(x)

    def get_config(self):
        config = super().get_config()
        config.update({"ratio": self.ratio})
        return config

# ================================
# MRI VALIDATION FUNCTIONS - CRITICAL FOR SAFETY
# ================================
def detect_brain_tissue_area(img):
    """Calculate the ratio of brain tissue to total image area"""
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    # Otsu thresholding to separate brain tissue from background
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Remove small noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Calculate tissue area ratio
    total_pixels = gray.shape[0] * gray.shape[1]
    tissue_pixels = np.count_nonzero(cleaned)
    tissue_ratio = tissue_pixels / total_pixels
    
    return tissue_ratio, cleaned

def find_largest_contour(binary_mask):
    """Find the largest contour in a binary mask using OpenCV"""
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, {}
    
    # Find largest contour by area
    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)
    
    if area < 100:  # Too small to be meaningful
        return None, {}
    
    # Calculate shape properties using OpenCV
    perimeter = cv2.arcLength(largest_contour, True)
    
    # Calculate circularity (4*pi*area / perimeter^2)
    if perimeter > 0:
        circularity = 4 * np.pi * area / (perimeter ** 2)
    else:
        circularity = 0
    
    # Calculate bounding rectangle for aspect ratio
    x, y, w, h = cv2.boundingRect(largest_contour)
    aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 1
    
    # Calculate solidity (contour area / convex hull area)
    hull = cv2.convexHull(largest_contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0
    
    return largest_contour, {
        'area': area,
        'perimeter': perimeter,
        'circularity': circularity,
        'aspect_ratio': aspect_ratio,
        'solidity': solidity
    }

def detect_skin_and_natural_colors(img):
    """Detect skin tones and natural colors that indicate a photo, not MRI - IMPROVED"""
    if len(img.shape) == 2:
        # Grayscale image - less likely to be a photo
        return False, {'skin_ratio': 0, 'natural_colors': False, 'avg_saturation': 0}
    
    # Convert to HSV for better color analysis
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Define skin color ranges in HSV (wider range for better detection)
    lower_skin1 = np.array([0, 25, 40])
    upper_skin1 = np.array([25, 255, 255])
    lower_skin2 = np.array([155, 25, 40])  # Wrap-around hue
    upper_skin2 = np.array([180, 255, 255])
    
    # Create skin color mask
    skin_mask1 = cv2.inRange(hsv, lower_skin1, upper_skin1)
    skin_mask2 = cv2.inRange(hsv, lower_skin2, upper_skin2)
    skin_mask = cv2.bitwise_or(skin_mask1, skin_mask2)
    
    # Calculate skin pixel ratio
    total_pixels = img.shape[0] * img.shape[1]
    skin_pixels = np.count_nonzero(skin_mask)
    skin_ratio = skin_pixels / total_pixels
    
    # Check for high color variance AND saturation (typical in natural photos)
    # Medical colormaps have color but lower saturation than natural photos
    saturation = hsv[:, :, 1]
    avg_saturation = np.mean(saturation[saturation > 0]) if np.any(saturation > 0) else 0
    high_saturation = avg_saturation > 80  # Natural photos have high saturation
    
    color_std = np.std(img, axis=(0,1))
    high_color_variance = np.mean(color_std) > 35  # Increased threshold
    
    # MRI images should have low skin color content and shouldn't look like natural photos
    has_skin_tones = skin_ratio > 0.15  # More than 15% skin-colored pixels
    has_natural_colors = high_color_variance and high_saturation
    
    is_natural_photo = has_skin_tones or has_natural_colors
    
    return is_natural_photo, {
        'skin_ratio': skin_ratio,
        'color_variance': np.mean(color_std),
        'avg_saturation': avg_saturation,
        'natural_colors': has_natural_colors
    }

def detect_face_features(img):
    """Detect facial features that indicate this is a photo of a person - more specific"""
    try:
        # Use OpenCV's built-in face cascade classifier
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        
        # More strict face detection parameters to avoid false positives with brain scans
        faces = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.3,      # Larger scale factor (less sensitive)
            minNeighbors=8,       # More neighbors required (more strict)
            minSize=(40, 40),     # Larger minimum face size
            maxSize=(120, 120)    # Maximum face size
        )
        
        # Additional validation: check if detected "faces" have skin-like properties
        valid_faces = 0
        if len(faces) > 0:
            for (x, y, w, h) in faces:
                face_region = img[y:y+h, x:x+w] if len(img.shape) == 3 else gray[y:y+h, x:x+w]
                
                # Check if face region has natural photo characteristics
                if len(img.shape) == 3:
                    # Check for skin tones in HSV
                    hsv_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2HSV)
                    lower_skin = np.array([0, 20, 50])
                    upper_skin = np.array([20, 255, 255])
                    skin_mask = cv2.inRange(hsv_face, lower_skin, upper_skin)
                    skin_ratio = np.count_nonzero(skin_mask) / (w * h)
                    
                    # Also check for typical photo texture/variance
                    face_std = np.std(face_region)
                    
                    # Only count as face if it has skin-like properties
                    if skin_ratio > 0.15 and face_std > 20:
                        valid_faces += 1
                else:
                    # For grayscale images, medical scans shouldn't have face-like intensity patterns
                    # Check intensity distribution - faces have different patterns than brain tissue
                    face_mean = np.mean(face_region)
                    face_std = np.std(face_region)
                    
                    # Medical brain tissue has different intensity characteristics than faces
                    # Brain tissue: lower contrast, specific intensity ranges
                    # Face in photo: higher contrast, broader intensity range
                    if face_std > 35 and 60 < face_mean < 200:  # Typical face characteristics
                        valid_faces += 1
        
        has_actual_faces = valid_faces > 0
        
        return has_actual_faces, {
            'detected_rectangles': len(faces),
            'valid_faces': valid_faces,
            'face_boxes': faces.tolist()
        }
        
    except Exception as e:
        return False, {'error': str(e)}

def check_medical_intensity_patterns(img):
    """Check for intensity patterns typical of medical imaging - improved for MRI"""
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    # Medical images typically have:
    # 1. Distinct tissue regions with different intensities
    # 2. Smooth gradients within tissue regions  
    # 3. Clear boundaries between different structures
    
    # Calculate histogram
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = hist.flatten() / hist.sum()
    
    # Check for medical image characteristics
    # 1. Should have significant dark background (air/background)
    background_peak = np.max(hist[0:40])  # Extended background range
    
    # 2. Should have tissue peaks in medical intensity ranges
    tissue_peaks = []
    for i in range(30, 220):  # Medical tissue intensity range
        if (hist[i] > hist[i-1] and hist[i] > hist[i+1] and 
            hist[i] > 0.005):  # Lower threshold for peaks
            tissue_peaks.append((i, hist[i]))
    
    # 3. Contrast should be in medical range
    tissue_contrast = gray.max() - gray.min()
    
    # 4. Check for circular/oval structures (more lenient)
    circles = cv2.HoughCircles(
        gray, 
        cv2.HOUGH_GRADIENT, 
        dp=1, 
        minDist=20,           # Closer circles allowed
        param1=40,            # Lower edge threshold
        param2=25,            # Lower accumulator threshold  
        minRadius=15,         # Smaller minimum radius
        maxRadius=80          # Reasonable maximum for brain structures
    )
    has_circular_structures = circles is not None and len(circles[0]) > 0
    
    # 5. Check intensity distribution characteristics of medical scans
    # Medical scans have smoother intensity transitions
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    has_medical_smoothness = 50 < laplacian_var < 2000  # Medical range
    
    # More lenient criteria for medical validation
    has_background = background_peak > 0.03   # Some background
    has_tissue_structure = len(tissue_peaks) >= 1  # At least one tissue peak
    reasonable_contrast = 40 < tissue_contrast < 220  # Broader range
    
    # Pass if most medical characteristics are present
    medical_checks = [
        has_background,
        has_tissue_structure, 
        reasonable_contrast,
        has_circular_structures,
        has_medical_smoothness
    ]
    
    # Pass if at least 3 out of 5 medical characteristics are present
    medical_pattern_valid = sum(medical_checks) >= 3
    
    return medical_pattern_valid, {
        'background_peak': background_peak,
        'num_tissue_peaks': len(tissue_peaks),
        'tissue_contrast': tissue_contrast,
        'has_circular_structures': has_circular_structures,
        'laplacian_variance': laplacian_var,
        'medical_checks_passed': sum(medical_checks)
    }

def strict_mri_validation(img_array):
    """
    Validation to accept brain MRI scans (including colorized ones), reject non-medical images
    Returns: (is_valid, validation_details, confidence_score)
    """
    validation_results = {}
    rejection_reasons = []
    warning_flags = []
    
    # Resize for analysis
    img = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    
    # Test 1: Reject photos with skin tones and natural colors (CRITICAL - must pass)
    is_photo, photo_details = detect_skin_and_natural_colors(img)
    validation_results['photo_detection'] = {
        'is_photo': is_photo,
        'rejected': is_photo,
        **photo_details
    }
    if is_photo:
        rejection_reasons.append("Contains skin tones/natural colors typical of photos")
    
    # Test 2: Reject images with faces (CRITICAL - must pass)
    has_faces, face_details = detect_face_features(img)
    validation_results['face_detection'] = {
        'has_faces': has_faces,
        'rejected': has_faces,
        **face_details
    }
    if has_faces:
        rejection_reasons.append("Contains human faces")
    
    # Test 3: Check for medical intensity patterns (IMPORTANT - but more lenient)
    has_medical_patterns, medical_details = check_medical_intensity_patterns(img)
    validation_results['medical_patterns'] = {
        'has_medical_patterns': has_medical_patterns,
        'rejected': not has_medical_patterns,
        **medical_details
    }
    if not has_medical_patterns:
        # Only warn, don't reject - some processed MRIs might not match all patterns
        warning_flags.append("Some medical imaging patterns not detected")
    
    # Test 4: Check tissue area (reasonable range for brain MRI)
    tissue_ratio, binary_mask = detect_brain_tissue_area(img)
    # Very lenient range: MRI scans can have varying amounts of visible brain tissue
    tissue_valid = 0.03 < tissue_ratio < 0.85  # Very broad range: 3-85%
    validation_results['tissue_analysis'] = {
        'tissue_ratio': tissue_ratio,
        'tissue_valid': tissue_valid,
        'rejected': not tissue_valid
    }
    if not tissue_valid:
        rejection_reasons.append(f"Tissue ratio {tissue_ratio:.2%} outside expected range (3-85%)")
    
    # Test 5: Color check - REMOVED the strict grayscale requirement
    # Accept both grayscale AND colorized medical images (like heat maps, colored overlays)
    if len(img.shape) == 3:
        b, g, r = cv2.split(img)
        color_diff = np.mean(np.abs(r.astype(int) - g.astype(int)) + 
                            np.abs(g.astype(int) - b.astype(int)) + 
                            np.abs(b.astype(int) - r.astype(int)))
        is_grayscale_like = color_diff < 15
        is_medical_colormap = 15 <= color_diff < 60  # Medical colormaps have moderate color differences
        
        color_acceptable = is_grayscale_like or is_medical_colormap
        
        validation_results['color_check'] = {
            'is_grayscale': is_grayscale_like,
            'is_medical_colormap': is_medical_colormap,
            'color_difference': color_diff,
            'color_acceptable': color_acceptable,
            'rejected': not color_acceptable
        }
        
        if not color_acceptable:
            rejection_reasons.append(f"Color pattern not typical of medical images (diff: {color_diff:.1f})")
        elif is_medical_colormap:
            warning_flags.append("Image appears to be a colorized/processed MRI scan")
    else:
        # Grayscale image - always acceptable
        validation_results['color_check'] = {
            'is_grayscale': True,
            'is_medical_colormap': False,
            'color_difference': 0,
            'color_acceptable': True,
            'rejected': False
        }
    
    # DECISION LOGIC: Must pass CRITICAL tests, tissue ratio check
    # Medical pattern check is advisory only
    critical_tests_passed = not is_photo and not has_faces and tissue_valid
    
    if len(img.shape) == 3:
        critical_tests_passed = critical_tests_passed and validation_results['color_check']['color_acceptable']
    
    is_valid_mri = critical_tests_passed
    
    # Calculate confidence score
    tests_passed = sum([
        not is_photo,  # Critical
        not has_faces,  # Critical
        has_medical_patterns,  # Advisory
        tissue_valid,  # Critical
        validation_results['color_check'].get('color_acceptable', True)  # Critical for color images
    ])
    
    confidence_score = tests_passed / 5.0
    
    validation_results['overall'] = {
        'is_valid': is_valid_mri,
        'confidence': confidence_score,
        'rejection_reasons': rejection_reasons,
        'warning_flags': warning_flags,
        'tests_passed': tests_passed,
        'total_tests': 5,
        'critical_tests_passed': critical_tests_passed
    }
    
    return is_valid_mri, validation_results, confidence_score

# ================================
# Preprocessing Functions
# ================================
def skull_strip(img):
    """Apply skull stripping to remove background"""
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    return cv2.bitwise_and(img, img, mask=mask)

def z_score_normalize(img):
    """Apply Z-score normalization"""
    if len(img.shape) == 3:
        mean = np.mean(img, axis=(0, 1))
        std = np.std(img, axis=(0, 1))
        std[std == 0] = 1e-8
        normalized = (img - mean) / std
    else:
        mean = np.mean(img)
        std = np.std(img)
        if std == 0:
            std = 1e-8
        normalized = (img - mean) / std
    
    return normalized.astype(np.float32)

def apply_clahe(img):
    """Apply CLAHE enhancement"""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    
    if len(img.shape) == 3 and img.shape[2] == 3:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        cl = clahe.apply(l)
        enhanced = cv2.merge((cl, a, b))
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    else:
        return clahe.apply(img)

def preprocess_image(img_array):
    """Complete preprocessing pipeline"""
    img = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    
    # Ensure RGB format
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif len(img.shape) == 3:
        if img.shape[2] == 1:
            img = cv2.cvtColor(img[:,:,0], cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        elif img.shape[2] == 3:
            pass  # Already RGB
    
    img = skull_strip(img)
    img = apply_clahe(img)
    img = z_score_normalize(img)
    
    img_min, img_max = img.min(), img.max()
    if img_max > img_min:
        img = ((img - img_min) / (img_max - img_min) * 255).astype(np.uint8)
    else:
        img = np.zeros_like(img, dtype=np.uint8)
    
    if img.shape != INPUT_SHAPE:
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif len(img.shape) == 3 and img.shape[2] != 3:
            if img.shape[2] == 1:
                img = cv2.cvtColor(img[:,:,0], cv2.COLOR_GRAY2RGB)
            else:
                img = img[:, :, :3]
    
    return img

# ================================
# Model Loading
# ================================
@st.cache_resource
def load_models():
    """Load all available models"""
    models = {}
    model_paths = {
        "ResNet50V2_CBAM": "C:/Users/ganes/Downloads/ibm_project2/models/resnet50v2_cbam_final.keras",
        "MobileNetV2_CBAM": "C:/Users/ganes/Downloads/ibm_project2/models/mobilenetv2_cbam_final.keras", 
        "DenseNet121_CBAM": "C:/Users/ganes/Downloads/ibm_project2/models/densenet121_cbam_final.keras",
        "EfficientNetB0_CBAM": "C:/Users/ganes/Downloads/ibm_project2/models/efficientnetb0_cbam_final.keras"
    }
    
    custom_objects = {
        'SpatialAttention': SpatialAttention,
        'CBAM': CBAM
    }
    
    for model_name, model_path in model_paths.items():
        try:
            if os.path.exists(model_path):
                model = tf.keras.models.load_model(
                    model_path, 
                    custom_objects=custom_objects,
                    compile=False
                )
                
                model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                    loss='categorical_crossentropy',
                    metrics=['accuracy']
                )
                
                models[model_name] = model
                st.sidebar.success(f"✅ {model_name} loaded")
            else:
                st.sidebar.error(f"❌ {model_path} not found")
        except Exception as e:
            st.sidebar.error(f"❌ Error loading {model_name}: {str(e)}")
    
    return models

# ================================
# Main Application
# ================================
def main():
    st.markdown('<h1 class="main-header">🧠 Brain Tumor Classification System</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("🔧 System Configuration")
    st.sidebar.markdown("---")
    
    # System info
    st.sidebar.subheader("📊 System Information")
    st.sidebar.info(f"TensorFlow: {tf.__version__}")
    st.sidebar.info(f"Input Shape: {INPUT_SHAPE}")
    st.sidebar.info(f"Classes: {len(CLASS_NAMES)}")
    
    # Load models
    models = load_models()
    
    if not models:
        st.error("❌ No models available. Please ensure model files are in the correct directory.")
        return
    
    # Class information
    st.sidebar.markdown("---")
    st.sidebar.subheader("📋 Classification Classes")
    for i, class_name in enumerate(CLASS_NAMES):
        st.sidebar.markdown(f"**{i}:** {class_name.title()}")
    
    # Validation settings
    st.sidebar.markdown("---")
    st.sidebar.subheader("🛡️ MRI Validation Settings")
    st.sidebar.success("✅ Accepts grayscale & colorized MRI scans")
    st.sidebar.info("System rejects photos, faces, and non-medical images")
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Upload Brain MRI Scan")
        st.warning("⚠️ **IMPORTANT**: Upload brain MRI scans (grayscale or colorized). Photos and non-medical images will be rejected.")
        st.info("ℹ️ **Accepted formats**: Standard grayscale MRI, colorized/processed MRI scans, T1/T2 weighted images")
        
        uploaded_file = st.file_uploader(
            "Choose a brain MRI scan image...",
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
            help="Upload a brain MRI scan for tumor classification. Other image types will be automatically rejected."
        )
        
        if uploaded_file is not None:
            # Display original image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Convert to numpy array
            img_array = np.array(image)
            if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            elif len(img_array.shape) == 3 and img_array.shape[2] == 4:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
            
            # Validate image button
            if st.button("🔍 Validate & Classify", type="primary"):
                with st.spinner("🔍 Performing MRI validation..."):
                    # CRITICAL: Strict validation - only accept MRI scans
                    is_valid_mri, validation_details, confidence_score = strict_mri_validation(img_array)
                    
                    # Store validation results
                    st.session_state.validation_results = validation_details
                    st.session_state.is_valid_mri = is_valid_mri
                    st.session_state.validation_confidence = confidence_score
                    
                    if is_valid_mri:
                        # Image passed validation - proceed with classification
                        with st.spinner("✅ Valid brain MRI detected! Processing for tumor classification..."):
                            processed_img = preprocess_image(img_array)
                            model_input = processed_img.astype(np.float32) / 255.0
                            model_input = np.expand_dims(model_input, axis=0)
                            
                            predictions = {}
                            for model_name, model in models.items():
                                try:
                                    pred = model.predict(model_input, verbose=0)
                                    predictions[model_name] = pred[0]
                                except Exception as e:
                                    st.error(f"❌ {model_name}: {str(e)}")
                                    predictions[model_name] = None
                            
                            st.session_state.predictions = predictions
                            st.session_state.processed_img = processed_img
                    else:
                        # Image failed validation
                        st.session_state.predictions = None
                        st.session_state.processed_img = None

    with col2:
        # Show validation results
        if hasattr(st.session_state, 'validation_results'):
            st.subheader("🔍 Image Validation Results")
            
            validation_results = st.session_state.validation_results
            is_valid = st.session_state.is_valid_mri
            confidence = st.session_state.validation_confidence
            
            if is_valid:
                st.markdown(f"""
                <div class="validation-pass">
                    <h4>✅ Valid Brain MRI Detected</h4>
                    <p>This image passed {validation_results['overall']['tests_passed']}/5 validation tests.</p>
                    <p>Confidence Score: {confidence:.1%}</p>
                    <p>The image appears to be a legitimate medical brain scan.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                rejection_reasons = validation_results['overall']['rejection_reasons']
                reasons_html = "<br>".join([f"• {reason}" for reason in rejection_reasons])
                st.markdown(f"""
                <div class="validation-fail">
                    <h4>❌ Invalid Image - NOT a Brain MRI</h4>
                    <p><strong>Failed {5 - validation_results['overall']['tests_passed']}/5 validation tests</strong></p>
                    <p><strong>Rejection Reasons:</strong></p>
                    <p>{reasons_html}</p>
                    <p><strong>Please upload only brain MRI scan images.</strong></p>
                </div>
                """, unsafe_allow_html=True)
            
            # Show detailed validation results
            with st.expander("📊 Detailed Validation Analysis", expanded=False):
                for test_name, results in validation_results.items():
                    if test_name == 'overall':
                        continue
                    
                    # Handle different result structures
                    if 'rejected' in results:
                        status = "❌ REJECTED" if results['rejected'] else "✅ PASSED"
                    elif 'is_photo' in results:
                        status = "❌ PHOTO DETECTED" if results['is_photo'] else "✅ NOT A PHOTO"
                    elif 'has_faces' in results:
                        status = "❌ FACES DETECTED" if results['has_faces'] else "✅ NO FACES"
                    elif 'has_medical_patterns' in results:
                        status = "✅ MEDICAL PATTERNS" if results['has_medical_patterns'] else "⚠️ NO MEDICAL PATTERNS (Advisory)"
                    elif 'tissue_valid' in results:
                        status = "✅ VALID TISSUE RATIO" if results['tissue_valid'] else "❌ INVALID TISSUE RATIO"
                    elif 'color_acceptable' in results:
                        if results['color_acceptable']:
                            if results.get('is_grayscale'):
                                status = "✅ GRAYSCALE MRI"
                            elif results.get('is_medical_colormap'):
                                status = "✅ COLORIZED MRI (Acceptable)"
                            else:
                                status = "✅ COLOR ACCEPTABLE"
                        else:
                            status = "❌ UNNATURAL COLOR PATTERN"
                    else:
                        status = "❓ UNKNOWN"
                    
                    st.markdown(f"**{test_name.replace('_', ' ').title()}:** {status}")
                    
                    # Show relevant details
                    for key, value in results.items():
                        if key not in ['rejected', 'is_photo', 'has_faces', 'has_medical_patterns', 
                                      'tissue_valid', 'is_grayscale', 'is_medical_colormap', 'color_acceptable']:
                            if isinstance(value, float):
                                if 'ratio' in key or 'confidence' in key:
                                    st.write(f"  - {key}: {value:.2%}")
                                else:
                                    st.write(f"  - {key}: {value:.3f}")
                            elif isinstance(value, (int, bool)):
                                st.write(f"  - {key}: {value}")
                            elif isinstance(value, list) and len(value) > 0:
                                st.write(f"  - {key}: {len(value)} detected")
                    st.write("---")
                
                # Show warnings if any
                if validation_results['overall'].get('warning_flags'):
                    st.warning("⚠️ **Advisory Warnings:**")
                    for warning in validation_results['overall']['warning_flags']:
                        st.write(f"• {warning}")
            
            # Show predictions only if image is valid
            if is_valid and hasattr(st.session_state, 'predictions') and st.session_state.predictions:
                st.subheader("🎯 Tumor Classification Results")
                predictions = st.session_state.predictions
                
                # Individual model results
                for model_name, pred in predictions.items():
                    if pred is not None:
                        predicted_class = CLASS_NAMES[np.argmax(pred)]
                        confidence_pct = np.max(pred) * 100
                        
                        st.markdown(f"""
                        <div class="prediction-box">
                            <h4>{model_name}</h4>
                            <p><strong>Prediction:</strong> {predicted_class.title()}</p>
                            <p><strong>Confidence:</strong> {confidence_pct:.2f}%</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Ensemble prediction
                valid_preds = [pred for pred in predictions.values() if pred is not None]
                if valid_preds:
                    ensemble_pred = np.mean(valid_preds, axis=0)
                    ensemble_class = CLASS_NAMES[np.argmax(ensemble_pred)]
                    ensemble_confidence = np.max(ensemble_pred) * 100
                    
                    st.markdown(f"""
                    <div class="ensemble-box">
                        <h3>🏆 Final Diagnosis</h3>
                        <h2>{ensemble_class.title()}</h2>
                        <h4>Confidence: {ensemble_confidence:.2f}%</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Confidence scores table
                    st.subheader("📋 Detailed Confidence Scores")
                    table_data = []
                    for model_name, pred in predictions.items():
                        if pred is not None:
                            row = {"Model": model_name}
                            for i, class_name in enumerate(CLASS_NAMES):
                                row[class_name.title()] = f"{pred[i] * 100:.2f}%"
                            table_data.append(row)
                    
                    if table_data:
                        df = pd.DataFrame(table_data)
                        st.dataframe(df, use_container_width=True)

    # Footer with important disclaimers
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem; background: #f8f9fa; border-radius: 10px;">
        <h4>🧠 Advanced Brain Tumor Classification System</h4>
        <p><strong>⚠️ MEDICAL DISCLAIMER:</strong> This system is for research purposes only and should not be used for actual medical diagnosis.</p>
        <p><strong>🛡️ SAFETY FEATURES:</strong> Advanced validation ensures only brain MRI scans are classified.</p>
        <p><strong>🎨 ACCEPTS:</strong> Grayscale MRI, Colorized MRI, T1/T2 weighted images, Processed medical scans</p>
        <p><strong>🔬 TECHNOLOGY:</strong> ResNet50V2, MobileNetV2, DenseNet121, EfficientNetB0 with CBAM Attention</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()