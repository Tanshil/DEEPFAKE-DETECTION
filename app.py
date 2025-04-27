

# app.py

import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# --- Define the CNN Model ---
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, 1)  # Binary classification

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 32 * 32)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# --- Load the trained model ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)
model.load_state_dict(torch.load("deepfake_detector.pth", map_location=device))
model.eval()

# --- Define prediction function ---
def predict_image(image):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    image = image.to(device)

    with torch.no_grad():
        output = model(image).squeeze()
        prediction = torch.sigmoid(output)
        predicted_label = (prediction >= 0.5).float()

    return predicted_label.item()

# --- Streamlit App ---
st.set_page_config(page_title="Deepfake Detector", layout="centered")
st.title("🕵️‍♂️ Deepfake Image Detector")
st.subheader("Upload an image and check if it's Real or Deepfake!")
st.write("Made by Akshat and Tanshil 🙌")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("⏳ Classifying... Please wait...")

    label = predict_image(image)

    if label == 1:
        st.error("😵‍💫 The image is predicted as: **DEEPFAKE**")
        st.info("⚡ Tip: Increase dataset size, train more epochs for better accuracy!")
    else:
        st.success("😎 The image is predicted as: **REAL**")
        st.info("⚡ Tip: Increase dataset size, train more epochs for better accuracy!")

st.markdown("---")
st.caption("Note: This is a basic model for educational purposes. Accuracy is limited.")
