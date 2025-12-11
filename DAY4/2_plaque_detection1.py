# Plaque detection using OpenCV and Haar Cascades

import torch 
import torch.nn as nn # For building neural network models
from torchvision import models, transforms # For image transformations
from PIL import Image # For image handling
import cv2 # OpenCV for image processing
import numpy as np 
import streamlit as st # For web app interface
import os

#----------------------------
# Device set up 
#----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

#----------------------------
#Load Pre-Trained Model (ResNet18)
#----------------------------
model = models.resnet18(pretrained=True) # This loads a ResNet18 model pre-trained on ImageNet
#Replace final Layer for binary Classification
num_features = model.fc.in_features # this will give number of input features to final layer
model.fc = nn.Linear(num_features, 2) # 2 Classes, plaque, Healthy , 
model = model.to(device)
#Load the Trained weitghts if availabl)

model.eval() # Set model to evaluation mode

#----------------------------
#Image Transformation 
#
transform = transforms.Compose([
    transforms.Resize((224, 224)), # Resize to 224x224 as expected by ResNet
    transforms.ToTensor(), # Convert image to PyTorch tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # Normalize as per ImageNet
])

#----------------------------
#Prediction 
#----------------------------
def predict_image(image_path):
    image = Image.open(image_path).convert("RGB") # Open image and convert to RGB
    image = transform(image).unsqueeze(0).to(device) # Apply transformations and add batch dimension
    with torch.no_grad():
        outputs = model(image) # Get model predictions
        _, predicted = torch.max(outputs, 1) # Get the index of the max log-probability
    return "Plaque Detected" if predicted.item() == 1 else "Healthy"

#----------------------------
#Streamlit App
#----------------------------
st.title("Dental Plaque Detection")
st.write("Upload an image of teeth to detect dental plaque.")   

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file:
    st.image(uploaded_file, caption='Uploaded Image.', use_container_width=True)
    st.write("")
    st.write("Detecting...")


#Save the uploaded image to a temporary location
    image_path = "temp_image.jpg"
    with open(image_path, "wb") as f: 
        f.write(uploaded_file.getbuffer())  
    
    #predict the image  
    result = predict_image(image_path)
    st.write(f"Prediction: {result}")       
    st.write("Done!")

    os.remove(image_path)  # Clean up the temporary image file
st.write("Developed by Your PhilipsDevteam")