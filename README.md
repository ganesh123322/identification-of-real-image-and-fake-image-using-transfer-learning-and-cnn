🧠 Real vs AI-Generated Image Identification

Identification of Real and Fake Images using Transfer Learning and CNN

📌 Project Overview

With the rise of generative AI models such as GANs and diffusion networks, it has become increasingly difficult to distinguish synthetic (AI-generated) images from real human faces.
This project proposes a multi-model deep learning framework that classifies images as Real or AI-Generated using a combination of CNN and transfer learning architectures deployed in a Streamlit web app for real-time detection.

🚀 Features

🧩 Deep Learning Models Used

Custom CNN

ResNet50V2

DenseNet121

EfficientNetB0

MobileNet

🧠 Technologies

TensorFlow / Keras

NumPy, OpenCV, PIL

Streamlit for deployment

Matplotlib and Pandas for visualization

📷 Web Interface

Upload an image (.jpg, .jpeg, .png, .bmp)

Get real-time predictions from multiple models

Display confidence levels, prediction scores, and ensemble consensus

GPU/CPU auto-configuration

⚙️ Installation
1️⃣ Clone the Repository
git clone https://github.com/ganesh123322/identification-of-real-image-and-fake-image-using-transfer-learning-and-cnn.git
cd identification-of-real-image-and-fake-image-using-transfer-learning-and-cnn

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Run the App
streamlit run app.py


Then open the link provided in your terminal (usually http://localhost:8501).

🧩 Model Details
Model	Type	Input Size	Accuracy
Custom CNN	Baseline	224×224	0.983
ResNet50V2	Transfer Learning	224×224	0.997
DenseNet121	Transfer Learning	224×224	0.998
EfficientNetB0	Transfer Learning	240×240	0.995
MobileNet	Transfer Learning	224×224	0.992

DenseNet121 achieved the highest validation accuracy of 99.8%.
These values are based on experimental results from the proposed study.

🧠 Architecture & Workflow

Image Upload → user uploads face image.

Preprocessing → resizing, normalization, and model-specific preprocessing (ResNet, VGG, EfficientNet, etc.).

Prediction → image fed through multiple deep learning models.

Ensemble Voting → final prediction decided based on consensus among models.

Visualization → displays bar chart (model vs. score) and pie chart (confidence distribution).

🧪 Dataset

Real and AI-generated faces sourced from Kaggle datasets and manually balanced.

Split: 70% training, 15% validation, 15% testing.

Augmentations: rotation (±20°), zoom (10%), horizontal flip, and brightness variation.

🧰 Key Python Libraries
import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

💻 Preprocessing Pipeline

CLAHE (Contrast Limited Adaptive Histogram Equalization)

Bilateral Filtering for denoising

Rescaling / normalization

Exact replication of Keras ImageDataGenerator preprocessing for consistency

🎯 Results Summary
Metric	Best Model (DenseNet121)
Accuracy	99.8%
Precision	99.7%
Recall	99.6%
F1-Score	99.7%
False Positive Rate	0.3%
False Negative Rate	0.4%

The ensemble approach further improved reliability across different samples. u can use based on accuracy or using voting or averaging its upto u.

🌐 Streamlit Interface Preview

Displays uploaded image with metadata (size, mode).

Shows individual model predictions and confidence.

Produces consensus output:

✅ Real Image

❌ AI-Generated Image

🔮 Future Work

Integration of Vision Transformers (ViT) and Swin Transformers

Support for video-based deepfake detection

Inclusion of Explainable AI 

Optimization for mobile/edge deployment

📚 References

Based on the IBM academic study:

“Real vs AI-Generated Image Identification with Enhanced Resolution and Multi-Model Deep Learning Approach”
Kalasalingam Academy of Research and Education, India (2025).

👨‍💻 Author

Ganesh D.
