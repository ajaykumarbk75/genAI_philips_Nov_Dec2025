#Plaque detection using Webcam

import torch
import torch.nn as nn # For building neural network models
from torchvision import models, transforms # For image transformations
import cv2 # OpenCV for image processing
import numpy as np
from PIL import Image # For image handling
import os
import streamlit as st # For web app interface

# #--------------
#Device Set up 
#--------------

device = "cuda" if torch.cuda.is_available() else "cpu"

#Load the Pretrainined Model (ResNet18)
model = models.resnet18(pretrained=True) # This loads a ResNet18 model pre
num_features = model.fc.in_features # this will give number of input features to final layer
model.fc = nn.Linear(num_features, 2) # 2 Classes, plaque, Healthy , 
model = model.to(device)

# Load your trained weights if available
model.eval() # Set model to evaluation mode

#----------------------------
#Image Transformation 
#----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)), # Resize to 224x224 as expected by ResNet
    transforms.ToTensor(), # Convert image to PyTorch tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # Normalize as per ImageNet
])  
#----------------------------
#Prediction function 
#----------------------------
def predict_frame(frame):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) 
    img_t = transform(image)
    batch_t = torch.unsqueeze(img_t, 0).to(device)
    with torch.no_grad():
        outputs = model(batch_t) # Get model predictions
        _, predicted = torch.max(outputs, 1) # Get the index of the max log-probability
    return "Plaque Detected" if predicted.item() == 1 else "Healthy"

#----------------------------
#Streamlit App
#----------------------------
st.title("Dental Plaque Detection via Webcam")
st.write("Click on 'Start Webcam' to detect dental plaque in real-time.")   

#use streamlit camera input
uploaded_file = st.camera_input("Capture your dental image", key="camera_input")

if uploaded_file is not None:
    # Convert uploaded image to OpneCV format
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, 1)

    #Make predictions 
pred = predict_frame(frame) # Predict plaque or healthy 

    #Display results
cv2.putText(frame, f"Prediction: {pred}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2) # put text on the image 
st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption='Webcam Image with Prediction', use_container_width=True)
st.write(f"Prediction: {pred}")
st.write("Note: Ensure your webcam is enabled and has permission to be accessed by this application.")
st.write("Developed By PhilipsDevteam")