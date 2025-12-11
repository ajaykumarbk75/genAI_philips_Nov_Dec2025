"""
🦷 Advanced Dental Plaque Detection AI
========================================

"""

import os
import uuid
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import cv2
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# ------------------------------
# Configuration
# ------------------------------
CLASS_NAMES = ["Plaque Detected", "Healthy"]
CONFIDENCE_THRESHOLD = 0.6
MODEL_PATH = "models/pretrained_model.pth"

# ------------------------------
# Device setup
# ------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
st.sidebar.info(f"🖥️ Running on: **{device.upper()}**")

# ------------------------------
# Model Architecture (Optional ResNet50)
# ------------------------------
class ImprovedPlaqueDetector(nn.Module):
    """Enhanced ResNet50 with dropout for better generalization"""
    def __init__(self, num_classes=2):
        super(ImprovedPlaqueDetector, self).__init__()
        self.backbone = models.resnet50(pretrained=True)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    def forward(self, x):
        return self.backbone(x)

# ------------------------------
# Load Model with caching
# ------------------------------
@st.cache_resource
def load_model():
    try:
        model = ImprovedPlaqueDetector(num_classes=2)
        model = model.to(device)
        if os.path.exists(MODEL_PATH):
            state_dict = torch.load(MODEL_PATH, map_location=device)
            model.load_state_dict(state_dict)
            st.sidebar.success("✅ Loaded fine-tuned model")
        else:
            st.sidebar.warning("⚠️ Using pretrained model (not fine-tuned)")
        model.eval()
        return model
    except Exception as e:
        st.sidebar.error(f"⚠️ Error: {e}. Using base ResNet18 model.")
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, 2)
        model = model.to(device)
        model.eval()
        return model

model = load_model()

# ------------------------------
# Dental Image Validation
# ------------------------------
def is_dental_image(image):
    """
    Strict validation if image contains teeth/dental content.
    Uses multiple detection methods to prevent false positives.
    Returns: (is_dental: bool, confidence: float, reason: str)
    """
    img_np = np.array(image) # Convert to numpy array
    h, w = img_np.shape[:2] # Get dimensions
    
    # 1. Color Analysis - dental images have very specific color ranges
    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
    
    # Teeth colors (white to yellowish-white)
    # More strict range for teeth
    lower_tooth = np.array([0, 0, 180])  # Very bright, low saturation
    upper_tooth = np.array([25, 80, 255])
    tooth_mask = cv2.inRange(hsv, lower_tooth, upper_tooth)
    tooth_ratio = np.sum(tooth_mask > 0) / tooth_mask.size
    
    # Pink/Red gums - strict range
    lower_gum1 = np.array([0, 40, 100])
    upper_gum1 = np.array([15, 255, 255])
    gum_mask1 = cv2.inRange(hsv, lower_gum1, upper_gum1) # Pinkish regions
    
    lower_gum2 = np.array([160, 40, 100])
    upper_gum2 = np.array([180, 255, 255])
    gum_mask2 = cv2.inRange(hsv, lower_gum2, upper_gum2) # Reddish regions
    
    gum_mask = cv2.bitwise_or(gum_mask1, gum_mask2)
    gum_ratio = np.sum(gum_mask > 0) / gum_mask.size #  Calculate gum ratio
    
    # 2. Detect mouth cavity (dark regions typical in dental photos)
    lower_dark = np.array([0, 0, 0])
    upper_dark = np.array([180, 255, 80])
    dark_mask = cv2.inRange(hsv, lower_dark, upper_dark)
    dark_ratio = np.sum(dark_mask > 0) / dark_mask.size
    
    # 3. Edge detection - teeth have very specific edge patterns
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 30, 100)
    edge_density = np.sum(edges > 0) / edges.size
    
    # 4. Horizontal edge analysis (teeth boundaries are often horizontal)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    horizontal_edges = np.abs(sobely) > np.abs(sobelx)
    horiz_ratio = np.sum(horizontal_edges) / horizontal_edges.size
    
    # 5. Brightness distribution (dental photos have specific lighting)
    avg_brightness = np.mean(gray)
    brightness_std = np.std(gray)
    
    # 6. Aspect ratio check (dental photos are usually landscape or square)
    aspect_ratio = w / h
    
    # 7. Detect fur/hair texture (reject animals)
    # Dental images have smooth regions, animal fur has high frequency texture
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    texture_variance = np.var(laplacian)
    
    # 8. White balance check (dental photos usually well-balanced)
    r_mean = np.mean(img_np[:,:,0])
    g_mean = np.mean(img_np[:,:,1])
    b_mean = np.mean(img_np[:,:,2])
    color_balance = max(abs(r_mean - g_mean), abs(g_mean - b_mean), abs(r_mean - b_mean))
    
    # Scoring system (very strict)
    score = 0
    max_score = 0
    reasons = []
    
    # CRITICAL: Must have significant tooth-colored regions (15-65%)
    max_score += 4
    if 0.15 <= tooth_ratio <= 0.65:
        score += 4
        reasons.append(f"✓ Tooth regions detected: {tooth_ratio*100:.1f}%")
    else:
        reasons.append(f"✗ No teeth detected: {tooth_ratio*100:.1f}% (need 15-65%)")
    
    # CRITICAL: Must have some gum/lip colored regions (3-35%)
    max_score += 3
    if 0.03 <= gum_ratio <= 0.35:
        score += 3
        reasons.append(f"✓ Gum/lip tissue: {gum_ratio*100:.1f}%")
    else:
        reasons.append(f"✗ No gum tissue: {gum_ratio*100:.1f}% (need 3-35%)")
    
    # Dark regions (mouth cavity) - 5-40%
    max_score += 2
    if 0.05 <= dark_ratio <= 0.40:
        score += 2
        reasons.append(f"✓ Mouth cavity: {dark_ratio*100:.1f}%")
    else:
        reasons.append(f"✗ No mouth cavity: {dark_ratio*100:.1f}%")
    
    # Edge density for dental photos
    max_score += 2
    if 0.06 <= edge_density <= 0.30:
        score += 2
        reasons.append(f"✓ Edge density: {edge_density*100:.1f}%")
    else:
        reasons.append(f"✗ Wrong edge pattern: {edge_density*100:.1f}%")
    
    # Horizontal edges (teeth boundaries)
    max_score += 1
    if 0.45 <= horiz_ratio <= 0.65:
        score += 1
        reasons.append(f"✓ Horizontal edges: {horiz_ratio*100:.1f}%")
    
    # Brightness (dental photos are well-lit: 120-210)
    max_score += 2
    if 120 <= avg_brightness <= 210:
        score += 2
        reasons.append(f"✓ Good lighting: {avg_brightness:.1f}")
    else:
        reasons.append(f"✗ Poor lighting: {avg_brightness:.1f} (need 120-210)")
    
    # Brightness variation (should have some contrast but not too much)
    max_score += 1
    if 30 <= brightness_std <= 70:
        score += 1
        reasons.append(f"✓ Contrast: {brightness_std:.1f}")
    
    # Reject high texture variance (fur/hair)
    max_score += 2
    if texture_variance < 800:  # Dental images have smoother texture
        score += 2
        reasons.append(f"✓ Smooth surface: {texture_variance:.1f}")
    else:
        reasons.append(f"✗ Fur/hair texture detected: {texture_variance:.1f}")
    
    # Aspect ratio (dental photos rarely extreme)
    max_score += 1
    if 0.5 <= aspect_ratio <= 2.5:
        score += 1
    else:
        reasons.append(f"✗ Unusual aspect ratio: {aspect_ratio:.2f}")
    
    # Color balance (dental photos are usually well-balanced)
    max_score += 1
    if color_balance < 40:
        score += 1
        reasons.append(f"✓ Color balanced")
    
    # STRICT THRESHOLD: Need at least 70% score
    confidence = score / max_score
    is_dental = score >= (max_score * 0.70)  # 70% threshold
    
    # Extra check: Must pass BOTH tooth and gum detection
    has_teeth = 0.15 <= tooth_ratio <= 0.65
    has_tissue = 0.03 <= gum_ratio <= 0.35
    
    if not (has_teeth and has_tissue):
        is_dental = False
        reasons.insert(0, "✗ FAILED: Missing teeth AND/OR gum tissue")
    
    reason_text = "\n".join(reasons)
    return is_dental, confidence, reason_text

# ------------------------------
# Image Preprocessing
# ------------------------------
def preprocess_dental_image(image):
    """Apply adaptive histogram equalization for dental images"""
    img_np = np.array(image)
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
    return Image.fromarray(enhanced)

# ------------------------------
# Transformations
# ------------------------------
base_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

tta_transforms = [
    base_transform,
    transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.functional.hflip,
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
]

# ------------------------------
# Prediction Function
# ------------------------------
def predict_with_confidence(image_path, use_tta=True):
    image = Image.open(image_path).convert("RGB")
    enhanced_image = preprocess_dental_image(image)
    all_predictions = []
    
    if use_tta:
        for transform in tta_transforms:
            if callable(transform):  # Handle functional transforms
                img_t = transform(enhanced_image)
            else:
                img_t = transform(enhanced_image)
            batch_t = torch.unsqueeze(img_t, 0).to(device)
            with torch.no_grad():
                output = model(batch_t)
                probs = F.softmax(output, dim=1)
                all_predictions.append(probs.cpu().numpy())
        avg_probs = np.mean(all_predictions, axis=0)[0]
    else:
        img_t = base_transform(enhanced_image)
        batch_t = torch.unsqueeze(img_t, 0).to(device)
        with torch.no_grad():
            output = model(batch_t)
            probs = F.softmax(output, dim=1)
            avg_probs = probs.cpu().numpy()[0]
    
    predicted_class = np.argmax(avg_probs)
    confidence = avg_probs[predicted_class]
    return predicted_class, confidence, avg_probs, enhanced_image

# ------------------------------
# Visualization
# ------------------------------
def create_confidence_chart(probabilities):
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ['#ff4444' if i==0 else '#44ff44' for i in range(len(CLASS_NAMES))]
    bars = ax.bar(CLASS_NAMES, probabilities * 100, color=colors, alpha=0.7)
    ax.set_ylabel('Confidence (%)')
    ax.set_ylim(0, 100)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{bar.get_height():.1f}%', ha='center', va='bottom')
    plt.tight_layout()
    return fig

def show_comparison(original, enhanced):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5))
    ax1.imshow(original)
    ax1.set_title('Original Image')
    ax1.axis('off')
    ax2.imshow(enhanced)
    ax2.set_title('Enhanced Image')
    ax2.axis('off')
    plt.tight_layout()
    return fig

# ------------------------------
# Streamlit App
# ------------------------------
st.title("🦷 Advanced Dental Plaque Detection AI")
st.markdown("Upload a clear image of teeth for instant analysis.")

# Sidebar
st.sidebar.header("⚙️ Settings")
use_tta = st.sidebar.checkbox("Use Test-Time Augmentation", value=True)
show_preprocessing = st.sidebar.checkbox("Show Preprocessing Steps", value=False)
validate_dental = st.sidebar.checkbox("Validate Dental Image", value=True, 
                                      help="Check if image contains teeth before analysis")

# Upload
uploaded_file = st.file_uploader("📤 Upload Dental Image", type=["jpg","jpeg","png"])
if uploaded_file:
    temp_filename = f"temp_{uuid.uuid4().hex}.jpg"
    with open(temp_filename, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Display original
    col1, col2 = st.columns(2)
    with col1:
        st.image(uploaded_file, caption="📸 Uploaded Image", use_container_width=True)
    
    # Prediction
    with st.spinner("🔍 Analyzing image..."):
        try:
            # First, validate if image is dental
            if validate_dental:
                test_img = Image.open(temp_filename).convert("RGB")
                is_dental, dental_conf, validation_reasons = is_dental_image(test_img)
                
                if not is_dental:
                    st.error("### ❌ Not a Dental Image")
                    st.warning(f"""
                    **This doesn't appear to be a dental image!**
                    
                    Validation Confidence: {dental_conf*100:.1f}%
                    
                    **Analysis:**
                    {validation_reasons}
                    
                    **Please upload:**
                    - Clear photos of teeth
                    - Dental X-rays
                    - Intraoral camera images
                    - Close-up tooth photographs
                    
                    **Avoid:**
                    - Photos of pets, animals, or people's faces
                    - Random objects or scenery
                    - Non-dental medical images
                    """)
                    if os.path.exists(temp_filename):
                        os.remove(temp_filename)
                    st.stop()
                else:
                    st.sidebar.success(f"✅ Dental image validated ({dental_conf*100:.1f}% confidence)")
            
            pred_class, conf, probs, enhanced_img = predict_with_confidence(
                temp_filename, use_tta=use_tta
            )
            
            if show_preprocessing:
                with col2:
                    st.image(enhanced_img, caption="✨ Enhanced Image", use_container_width=True)
            
            # Display results
            st.markdown("---")
            st.subheader("📊 Analysis Results")
            result_text = CLASS_NAMES[pred_class]
            if result_text == "Plaque Detected":
                st.error(f"### ⚠️ {result_text}")
            else:
                st.success(f"### ✅ {result_text}")
            
            # Confidence metric
            conf_pct = conf * 100
            st.metric(label="Confidence", value=f"{conf_pct:.1f}%")
            
            # Show confidence chart
            fig_conf = create_confidence_chart(probs)
            st.pyplot(fig_conf)
            plt.close(fig_conf)
            
            # Preprocessing comparison
            if show_preprocessing:
                original_img = Image.open(temp_filename)
                fig_comp = show_comparison(original_img, enhanced_img)
                st.pyplot(fig_comp)
                plt.close(fig_comp)
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
        finally:
            if os.path.exists(temp_filename):
                os.remove(temp_filename)

else:
    st.info("👆 Upload an image to get started.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align:center'>
<p><strong>⚕️ Disclaimer:</strong> This AI tool is for educational purposes. Consult a dentist for professional advice.</p>
<p>Developed by SK | Powered by PyTorch & ResNet</p>
</div>
""", unsafe_allow_html=True)
