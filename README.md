
# Deepfake Detector

### Project by TANSHIL TIGRAN  and AKSHAT TIWARI

This project aims to create a basic deepfake detector using Convolutional Neural Networks (CNNs). The model is designed to classify images as either real or deepfake. While the model is simple and serves as a starting point, the accuracy is limited due to the small dataset used for training and the low number of epochs.

---

## Overview
- **Dataset**: The dataset consists of 100 images each in two categories: "REAL" and "FAKE".
- **Model**: The model uses a basic CNN architecture.
- **Epochs**: The number of epochs is kept low to allow the model to run on any device, but they can be increased if a GPU is available for faster training.
- **Note**: Due to the small dataset, the accuracy is limited. To improve results, consider adding more data and increasing the number of epochs, especially if you have access to a powerful GPU.

---

## Features
- Basic binary classification (real vs. fake images).
- Model trained on a small dataset (100 images per category).
- Uses a simple CNN architecture with 2 convolutional layers and 2 fully connected layers.
- Output: "REAL" or "DEEPFAKE" based on the image prediction.
- Model training and evaluation included, with the option to save the trained model.

---

## Dependencies
Ensure you have the following libraries installed:
- `torch`
- `torchvision`
- `PIL`
- `numpy`

You can install the dependencies using:

```bash
pip install torch torchvision pillow
```

---

## Usage

### 1. Training
The model is trained on a small dataset with 100 images each in the "REAL" and "FAKE" categories. The training script includes the following:
- Data loading and preprocessing using `ImageFolder`.
- Model definition with a simple CNN.
- Training loop using binary cross-entropy loss.
- Save the trained model as `deepfake_detector.pth`.

### 2. Testing
After training, the model is tested on a separate test set, and the accuracy is printed to the console. The test accuracy depends on the dataset size and model architecture.

### 3. Prediction
You can use the trained model to predict whether an image is real or fake. To test the model, provide a test image path and call the `predict_image()` function:

```python
test_img_path = "path/to/your/image.jpg"
predict_image(test_img_path)
```

The model will output either REAL or DEEPFAKE based on the prediction.

---

## Code Explanation

### Model Definition (`SimpleCNN`):

```python
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, 1)  # Output layer for binary classification
```

### Training Loop:

```python
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
```

### Prediction:

```python
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
            print(f"The image '{image_path}' is predicted DEEPFAKE")
        else:
            print(f"The image '{image_path}' is predicted as: REAL")
```

---

## Improving the Model
- **Add more images** to the dataset: The model's performance will improve significantly with more data.
- **Increase the number of epochs**: If you have a GPU, you can increase the number of epochs to improve accuracy.
- **Use a better architecture**: You can experiment with deeper networks or pre-trained models (e.g., ResNet, VGG).

---

## Important Notes
- **Accuracy**: The model is not highly accurate due to the small dataset and low epochs. This is a basic prototype and should be refined for practical use.
- **Training Time**: On devices without a GPU, training might take time due to the small number of epochs.
- **GPU Usage**: The model can be trained on a GPU for faster performance. Simply adjust the number of epochs if you are training on a powerful device.

---

## Conclusion
This project serves as a basic deepfake detection model. While the performance may not be ideal due to the limited dataset and model complexity, it provides a starting point for further experimentation and improvement.

---

## Acknowledgements
- This project was created as part of the deepfake detection initiative by Akshat GF202217215 and Tanshil.

---
