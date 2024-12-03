import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from PIL import Image
import segmentation_models_pytorch as smp
import base64
import os

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

def add_custom_css(background_image_path):
    with open(background_image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode()

    st.markdown(
        f"""
        <style>
        /* Background Styling */
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
        /* Sidebar Styling */
        .css-1d391kg {{
            background: rgba(255, 255, 255, 0.8); /* Light background for the sidebar */
            border-radius: 10px;
        }}
        .css-1d391kg h1, .css-1d391kg h2, .css-1d391kg h3 {{
            color: black !important; /* Sidebar headings in black */
            font-weight: bold;
        }}
        .css-1d391kg p, .css-1d391kg ul {{
            color: #333333 !important; /* Sidebar text in dark gray */
        }}
        /* Main Content Headings */
        h1, h2, h3, label {{
            color: white !important;
            text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.7); /* Add shadow for better contrast */
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

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

# Apply custom CSS for background image and sidebar
add_custom_css("dronepic.png")

# Define explicit model paths
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

# Upload image
st.subheader("Upload an Image")
uploaded_image = st.file_uploader("", type=["jpg", "png"])

# Perform segmentation if an image is uploaded and a model is selected
if uploaded_image and model_name != "Select a Model":
    image = Image.open(uploaded_image).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    model = models[model_name]
    prediction = predict_image(image, model)

    # Map prediction to color image
    color_pred = map_class_to_color(prediction)

    st.image(color_pred, caption="Segmented Image", use_column_width=True)


