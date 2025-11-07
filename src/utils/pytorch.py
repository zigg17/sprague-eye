import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
import torchvision.transforms as transforms

from PIL import Image
import cv2
import numpy as np

import os

from tkinter import filedialog


class ResNet18BBoxPredictor(nn.Module):
    def __init__(self):
        super(ResNet18BBoxPredictor, self).__init__()
        # Use weights instead of pretrained=True
        self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 4)  # Predict 4 bbox coords

    def forward(self, x):
        return self.backbone(x)

def initialize_model(checkpoint_path):
    """
    Initialize the model, transformations, and device.

    Args:
        checkpoint_path (str): Path to the checkpoint file.

    Returns:
        model: The initialized model with loaded weights.
        transform: Transformations to be applied to input images.
        device: Device on which the model will run (CPU or CUDA).
    """
    # Initialize the model
    model = ResNet18BBoxPredictor()

    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Define the transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    

    return model, transform, device

def predict_and_mask(model, image_path, transform, device):
    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    input_image = transform(image).unsqueeze(0).to(device)  # Add batch dimension

    # Predict bounding box
    model.eval()
    with torch.no_grad():
        pred_bbox = model(input_image).squeeze().cpu().numpy()  # Remove batch and move to CPU

    # Convert bbox to integers
    x, y, w, h = map(int, pred_bbox)

    # Load the original image using cv2 for masking
    image_cv2 = cv2.imread(image_path)
    if image_cv2 is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Create a black mask
    mask = np.zeros_like(image_cv2)

    # Define ROI and copy it onto the mask
    roi = image_cv2[y:y+h, x:x+w]
    mask[y:y+h, x:x+w] = roi

    # Save the masked image, replacing the original image
    cv2.imwrite(image_path, mask)