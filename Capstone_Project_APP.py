import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from PIL import Image
import segmentation_models_pytorch as smp
import zipfile
import os
from pathlib import Path
import base64
import pickle
import pandas as pd
import cv2
import xlsxwriter
import shutil
from io import BytesIO

# Define your model architectures
class SimpleFCN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleFCN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1),
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# Initialize models based on their names
def initialize_model(model_name):
    if model_name == "SimpleFCN":
        return SimpleFCN(num_classes=24)
    elif model_name == "FPN":
        return smp.FPN(encoder_name="efficientnet-b3", encoder_weights="imagenet", in_channels=3, classes=24)
    elif model_name == "DeepLabV3Plus":
        return smp.DeepLabV3Plus(encoder_name="efficientnet-b3", encoder_weights="imagenet", in_channels=3, classes=24)
    elif model_name == "U-Net":
        return smp.Unet(encoder_name="mobilenet_v2", encoder_weights="imagenet", classes=24, activation=None)
    else:
        raise ValueError(f"Unknown model: {model_name}")


# Load models from paths
@st.cache_resource
def load_models(model_paths, model_name_mapping):
    models = {}
    for display_name, path in model_paths.items():
        internal_name = model_name_mapping[display_name]
        model = initialize_model(internal_name)

        # Attempt to load the model or state_dict
        state_dict = torch.load(path, map_location=torch.device("cpu"))
        if isinstance(state_dict, nn.Module):  # If the file contains the full model
            models[display_name] = state_dict.eval()
        elif isinstance(state_dict, dict):  # If the file contains a state_dict
            if any(key.startswith("module.") for key in state_dict.keys()):
                state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}
            model.load_state_dict(state_dict, strict=False)
            models[display_name] = model.eval()
        else:
            raise TypeError(f"Expected state_dict or model, got {type(state_dict)} for {path}")

    return models


# Define image preprocessing
def transform_image(image):
    transform = transforms.Compose([ 
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0)


# Predict function
def predict_image(image, model):
    image = transform_image(image)
    with torch.no_grad():
        output = model(image)
    output = torch.argmax(output.squeeze(), dim=0).cpu().numpy()
    return output


# Map predictions to colors for visualization
def map_class_to_color(prediction):
    class_to_color = {
        0: [0, 0, 0],         # Unlabeled
        1: [128, 64, 128],    # Paved Area
        2: [130, 76, 0],      # Dirt
        3: [0, 102, 0],       # Grass
        4: [112, 103, 87],    # Gravel
        5: [28, 42, 168],     # Water
        6: [48, 41, 30],      # Rocks
        7: [0, 50, 89],       # Pool
        8: [107, 142, 35],    # Vegetation
        9: [70, 70, 70],      # Roof
        10: [102, 102, 156],  # Wall
        11: [254, 228, 12],   # Window
        12: [254, 148, 12],   # Door
        13: [190, 153, 153],  # Fence
        14: [153, 153, 153],  # Fence Pole
        15: [255, 22, 96],    # Person
        16: [102, 51, 0],     # Dog
        17: [9, 143, 150],    # Car
        18: [119, 11, 32],    # Bicycle
        19: [51, 51, 0],      # Tree
        20: [190, 250, 190],  # Bald Tree
        21: [112, 150, 146],  # AR Marker
        22: [2, 135, 115],    # Obstacle
        23: [255, 0, 0],      # Conflicting
    }

    height, width = prediction.shape
    color_image = np.zeros((height, width, 3), dtype=np.uint8)

    for class_id, color in class_to_color.items():
        mask = prediction == class_id
        color_image[mask] = color

    return color_image


# Load bounding boxes (for person detection)
@st.cache_resource
def load_bounding_boxes(file_path):
    with open(file_path, "rb") as file:
        data = pickle.load(file)
    return data


bounding_boxes = load_bounding_boxes("imgIdToBBoxArray.p")


# Streamlit Interface
st.title("Drone Segmentation for Rescue and Defence")

# Add sidebar content
st.sidebar.title("About This App")
st.sidebar.markdown(
    """
    ### What Does This App Do?
    This application performs **semantic segmentation** on drone images using various state-of-the-art deep learning models.
    
    ### Why Is It Useful?
    - **Rescue Operations:** Quickly identify and segment areas like water, vegetation, and structures for effective rescue planning.
    - **Defence Applications:** Analyze aerial views for critical decision-making in defence operations.
    - **Urban Planning:** Use segmentation results for accurate mapping and planning in urban areas.
    
    ### How to Use
    1. **Select a Model:** Choose a segmentation model from the dropdown menu.
    2. **Upload an Image:** Drag and drop or upload a drone image.
    3. **View Results:** See the segmented output with color-coded classes.
    """
)

# Model paths and mapping
model_paths = {
    "U-Net (Accuracy: 0.81)": "Unet-Mobilenet.pt",
    "SimpleFCN (Accuracy: 0.48)": "SimpleFCN_best_model.pth",
    "FPN (Accuracy: 0.65)": "FPN_best_model.pth",
    "DeepLabV3Plus (Accuracy: 0.69)": "DeepLabV3Plus_best_model.pth",
}

# Map display names to internal names
model_name_mapping = {
    "U-Net (Accuracy: 0.81)": "U-Net",
    "SimpleFCN (Accuracy: 0.48)": "SimpleFCN",
    "FPN (Accuracy: 0.65)": "FPN",
    "DeepLabV3Plus (Accuracy: 0.69)": "DeepLabV3Plus",
}

# Load models
models = load_models(model_paths, model_name_mapping)

# Model selection
st.subheader("Select a Model")
model_name = st.selectbox("", ["Select a Model"] + list(models.keys()))

# Upload folder for batch processing
st.subheader("Batch Processing")
uploaded_folder = st.file_uploader("Upload a zip folder of images", type="zip")

if st.button("Process Images"):
    if uploaded_folder:
        # Clear the previous images in temp_images directory
        temp_dir = Path("temp_images")
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        temp_dir.mkdir(exist_ok=True)

        # Open the zip file and extract its contents
        with zipfile.ZipFile(uploaded_folder, "r") as zip_ref:
            zip_ref.extractall(temp_dir)

        # List all image files in the extracted folder (including subdirectories)
        image_files = list(temp_dir.glob('**/*.jpg')) + list(temp_dir.glob('**/*.png')) + list(temp_dir.glob('**/*.jpeg'))
        st.write(f"Uploaded folder contains {len(image_files)} image files.")

        # Process each image in the folder
        if len(image_files) > 0:
            results = []
            excel_path = "batch_results.xlsx"

            with xlsxwriter.Workbook(excel_path) as workbook:
                worksheet = workbook.add_worksheet("Results")
                worksheet.write(0, 0, "Image Name")
                worksheet.write(0, 1, "Persons Detected")
                worksheet.write(0, 2, "Processed Image")

                row = 1
                for image_path in image_files:
                    image = Image.open(image_path).convert("RGB")
                    model = models[model_name]
                    prediction = predict_image(image, model)

                    # Map the prediction to a color image
                    color_pred = map_class_to_color(prediction)

                    # Convert the image to bytes for Excel insertion
                    image_stream = BytesIO()
                    Image.fromarray(color_pred).save(image_stream, format="PNG")
                    image_stream.seek(0)

                    # Generate file name and the number of persons (from bounding boxes)
                    image_name = image_path.stem
                    num_persons = len(bounding_boxes.get(image_name, []))  # Get number of persons from bounding boxes

                    # Write results to Excel
                    worksheet.write(row, 0, image_name)
                    worksheet.write(row, 1, num_persons)
                    worksheet.insert_image(row, 2, image_stream)

                    # Append results
                    results.append({
                        "Image Name": image_name,
                        "Persons Detected": num_persons
                    })

                    row += 1

            # Provide a download button for the Excel sheet
            st.download_button(
                label="Download Processed Results",
                data=open(excel_path, "rb").read(),
                file_name=excel_path,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

        else:
            st.write("No valid images found in the uploaded folder.")
