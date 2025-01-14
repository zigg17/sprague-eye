import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
import torchvision.transforms as transforms

from PIL import Image
import cv2
import numpy as np

import os

# Define Model
class ResNet18BBoxPredictor(nn.Module):
    def __init__(self):
        super(ResNet18BBoxPredictor, self).__init__()
        # Use weights instead of pretrained=True
        self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 4)  # Predict 4 bbox coords

    def forward(self, x):
        return self.backbone(x)

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
    
def images_to_video(image_folder, output_file, fps):
    """
    Convert a folder of images into a video.

    Parameters:
    - image_folder: Path to the folder containing images.
    - output_file: Path to the output video file (e.g., 'output.mp4').
    - fps: Frames per second for the output video.
    """
    # Get a sorted list of image files in the folder
    images = [img for img in os.listdir(image_folder) if img.endswith((".png", ".jpg", ".jpeg"))]
    images.sort()  # Ensure images are in the correct order

    if not images:
        print("No images found in the folder.")
        return

    # Read the first image to get dimensions
    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path)
    height, width, _ = frame.shape

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # For .mp4 output
    video = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    # Loop through the images and write them to the video
    for image in images:
        image_path = os.path.join(image_folder, image)
        frame = cv2.imread(image_path)
        video.write(frame)

    # Release the video writer
    video.release()
    print(f"Video saved as {output_file}")

# Initialize the model
model = ResNet18BBoxPredictor()

# Load the checkpoint
checkpoint_path = "models/bbox_cage.pth"
checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'), weights_only=True)

# Load the model state_dict
model.load_state_dict(checkpoint['model_state_dict'])

image_path = "/Users/jakeziegler/Desktop/PROJECTS/SPRAGUE-EYE/JSE-TRAINING/SAMPLED-FRAMES/frame_00031.jpg"

transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Folder path
folder_path = "/Users/jakeziegler/Desktop/PROJECTS/SPRAGUE-EYE/JSE-TRAINING/FRAMES/JSE2"

# Loop through all images in the folder
for filename in os.listdir(folder_path):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(folder_path, filename)
        try:
            predict_and_mask(model, image_path, transform, device)
            print(f"Processed and replaced: {filename}")
        except Exception as e:
            print(f"Error processing {filename}: {e}")