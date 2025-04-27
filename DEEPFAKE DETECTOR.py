#PROJECT BY AKSHAT GF202217215 AND TANSHIL THIS IS A VERY BASIC MODEL DESIGNED BY US TO DISCRIMINATE BETWEEN IMAGES THAT
#ARE DEEPFAKE OR NOT
#KEEPING THAT IN MIND THAT THE MODEL IS NOT ACCURATE AND THE TRAINING DATASET IS OF 100 IMAGES EACH IN "REAL" AND "FAKE"
#THE EPOCHS ARE LOW TO MAKE IT RUN ON ANY DEVICE
#EITHER WAY IF THERE IS A GPU THE EPOCHS CAN BE TWITCHED!
#MARKS DEDIJYEGA SIR PLEASE

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torch.utils.data import random_split
from PIL import Image
import os

# --- Check device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Define the CNN Model ---
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, 1)  # Binary classification (real vs fake)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 32 * 32)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleCNN().to(device)

# --- Data loading and preprocessing ---
data_dir = r"dataset"  # Your dataset folder

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

dataset = ImageFolder(root=data_dir, transform=transform)

# Split into train and test
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# --- Loss and optimizer ---
criterion = nn.BCEWithLogitsLoss()  # Good for binary classification
optimizer = optim.Adam(model.parameters(), lr=0.001)

# --- Training ---
num_epochs = 5

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device).float()

        optimizer.zero_grad()
        outputs = model(images).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}")

# --- Testing ---
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device).float()
        outputs = model(images).squeeze()
        preds = (torch.sigmoid(outputs) >= 0.5).float()
        total += labels.size(0)
        correct += (preds == labels).sum().item()

accuracy = 100 * correct / total
print(f"\nAccuracy of the model on test images: {accuracy:.2f}%")

# --- Saving the model ---
torch.save(model.state_dict(), "deepfake_detector.pth")
print("✅ Model saved as deepfake_detector.pth")

# --- Prediction function ---
def predict_image(image_path):
    model.eval()
    image = Image.open(image_path).convert('RGB')
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

        if predicted_label.item() == 1:
            print(f"\n The image '{image_path}' is predicted DEEPFAKE")
            print(f"if not the expected results use a good gpu and add more images to the dataset increase epochs")
        else:
            print(f"\n The image '{image_path}' is predicted as: REAL")
            print(f"if not the expected results use a good gpu and add more images to the dataset increase epochs")


# --- Predict on your own image ---
test_img_path = r"test2.jpg"  # Change this to your test image path
predict_image(test_img_path)