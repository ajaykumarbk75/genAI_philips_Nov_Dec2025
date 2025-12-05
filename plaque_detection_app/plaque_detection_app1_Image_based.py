# Device Setup
# The program checks if a GPU is available and uses it for faster model inference; otherwise, it uses the CPU.

# Load and Prepare the Model
# It loads a pretrained ResNet18 model, replaces the final layer to classify 2 categories (Plaque / Healthy), 
# and sets the model to evaluation mode.

# Image Preprocessing
# Incoming webcam images are resized, converted to tensors, and normalized to match the format ResNet18 expects.

# Prediction Function
# Each captured webcam frame is converted to a PIL image, processed through the model, and classified as “Plaque” or “Healthy”.

# Streamlit Web App
# Users capture a photo using their webcam.
# The app processes the image, performs prediction, overlays the result on the image, and displays it back in the browser.

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import cv2
import numpy as np
import streamlit as st

# ------------------------------
# Device setup
# ------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------
# Load Pre-trained Model (ResNet18)
# ------------------------------
model = models.resnet18(pretrained=True)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)
model = model.to(device)
# Load your trained weights if available
# model.load_state_dict(torch.load("models/pretrained_model.pth", map_location=device))
model.eval()

# ------------------------------
# Image Transformations
# ------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ------------------------------
# Prediction Function
# ------------------------------
def predict_frame(frame):
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img_t = transform(img)
    batch_t = torch.unsqueeze(img_t, 0).to(device)
    with torch.no_grad():
        out = model(batch_t)
        _, pred = torch.max(out, 1)
    return "Plaque" if pred.item() == 0 else "Healthy"

# ------------------------------
# Streamlit App
# ------------------------------
st.title("🦷 Live Dental Plaque Detection AI")
st.write("Use your webcam to detect plaque in real-time.")

# Use Streamlit camera input
uploaded_file = st.camera_input("Capture your dental image", key="camera_input") # Capture image from webcam

if uploaded_file is not None:
    # Convert uploaded image to OpenCV format
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, 1)
    
    # Make prediction
    pred = predict_frame(frame) #   Predict plaque or healthy
    
    # Display prediction on image
    cv2.putText(frame, f"Prediction: {pred}", (10, 30),   # here we put the text on the image
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if pred=="Healthy" else (0,0,255), 2) # Green for healthy, Red for plaque
    
    st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption=f"Prediction: {pred}", use_column_width=True) # Display image with prediction

st.write("Developed by SK")
