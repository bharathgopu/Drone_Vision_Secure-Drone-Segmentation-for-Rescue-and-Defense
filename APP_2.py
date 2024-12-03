import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from PIL import Image
import segmentation_models_pytorch as smp
import base64
import os
import pickle

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

# Apply custom CSS for background image and sidebar
def add_custom_css(background_image_path):
    with open(background_image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode()

    st.markdown(
        f"""
        <style>
        body {{
            background-image: url("data:image/png;base64,{base64_image}");
            background-size: cover;
            background-attachment: fixed;
            background-position: center;
            background-repeat: no-repeat;
        }}
        .stApp {{
            background-color: rgba(0, 0, 0, 0.5); /* Add transparency */
            border-radius: 10px;
        }}
        section[data-testid="stSidebar"] {{
            background: rgba(255, 255, 255, 0.8); /* Light background for the sidebar */
            border-radius: 10px;
            padding: 15px;
        }}
        section[data-testid="stSidebar"] h1, 
        section[data-testid="stSidebar"] h2, 
        section[data-testid="stSidebar"] h3 {{
            color: black !important; /* Sidebar headings in black */
            font-weight: bold;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

# Streamlit Interface
st.title("Drone Segmentation for Rescue and Defence")

# Apply custom CSS
add_custom_css("dronepic.png")

# Load bounding box data
@st.cache_resource
def load_bounding_boxes(file_path):
    with open(file_path, "rb") as file:
        data = pickle.load(file)
    return data

bounding_boxes = load_bounding_boxes("imgIdToBBoxArray.p")

# Define model paths and load models
model_paths = {
    "U-Net (Accuracy: 0.81)": "Unet-Mobilenet.pt",
    "SimpleFCN (Accuracy: 0.48)": "SimpleFCN_best_model.pth",
    "FPN (Accuracy: 0.65)": "FPN_best_model.pth",
    "DeepLabV3Plus (Accuracy: 0.69)": "DeepLabV3Plus_best_model.pth",
}

model_name_mapping = {
    "U-Net (Accuracy: 0.81)": "U-Net",
    "SimpleFCN (Accuracy: 0.48)": "SimpleFCN",
    "FPN (Accuracy: 0.65)": "FPN",
    "DeepLabV3Plus (Accuracy: 0.69)": "DeepLabV3Plus",
}

models = load_models(model_paths, model_name_mapping)

# Model selection
st.subheader("Select a Model")
model_name = st.selectbox("", ["Select a Model"] + list(models.keys()))

# Upload image
st.subheader("Upload an Image")
uploaded_image = st.file_uploader("", type=["jpg", "png"])

# Perform segmentation and count persons
if uploaded_image and model_name != "Select a Model":
    image = Image.open(uploaded_image).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    model = models[model_name]
    prediction = predict_image(image, model)

    # Extract image ID from file name (e.g., 000.jpg to 598.jpg)
    image_id = os.path.splitext(os.path.basename(uploaded_image.name))[0]
    if image_id in bounding_boxes:
        num_persons = len(bounding_boxes[image_id])
        st.write(f"Number of Persons Detected: {num_persons}")
    else:
        st.write("No bounding boxes available for this image.")
