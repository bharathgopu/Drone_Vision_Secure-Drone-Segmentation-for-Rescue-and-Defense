import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from PIL import Image, ImageDraw
import segmentation_models_pytorch as smp
import pickle
import os
import base64

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

        state_dict = torch.load(path, map_location=torch.device("cpu"))
        if isinstance(state_dict, nn.Module):
            models[display_name] = state_dict.eval()
        elif isinstance(state_dict, dict):
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
        15: [255, 22, 96],    # Person
    }
    height, width = prediction.shape
    color_image = np.zeros((height, width, 3), dtype=np.uint8)

    for class_id, color in class_to_color.items():
        mask = prediction == class_id
        color_image[mask] = color

    return color_image

# Draw bounding boxes on the segmented image
def draw_bounding_boxes(image, bboxes):
    draw = ImageDraw.Draw(image)
    for bbox in bboxes:
        x_min, y_min = map(int, bbox[0])
        x_max, y_max = map(int, bbox[1])
        draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=2)
    return image

# Load bounding boxes from Pickle file
@st.cache_resource
def load_bounding_boxes(pickle_file):
    with open(pickle_file, 'rb') as file:
        data = pickle.load(file)
    return data

# Load bounding boxes
bounding_boxes = load_bounding_boxes("imgIdToBBoxArray.p")

# Apply custom CSS
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
            background-color: rgba(0, 0, 0, 0.5); /* Transparency */
            border-radius: 10px;
        }}
        section[data-testid="stSidebar"] {{
            background: rgba(255, 255, 255, 0.8); /* Light background */
            border-radius: 10px;
        }}
        section[data-testid="stSidebar"] h1, 
        section[data-testid="stSidebar"] h2, 
        section[data-testid="stSidebar"] h3 {{
            color: black !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

# Streamlit Interface
st.title("Drone Segmentation for Rescue and Defence")

# Apply custom CSS
add_custom_css("dronepic.png")

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

if uploaded_image and model_name != "Select a Model":
    # Extract image ID from file name
    image_id = os.path.splitext(os.path.basename(uploaded_image.name))[0]

    # Perform segmentation
    image = Image.open(uploaded_image).convert("RGB")
    model = models[model_name]
    prediction = predict_image(image, model)
    color_pred = map_class_to_color(prediction)

    # Overlay bounding boxes
    if image_id in bounding_boxes:
        bboxes = bounding_boxes[image_id]
        segmented_image = draw_bounding_boxes(Image.fromarray(color_pred), bboxes)
        st.image(segmented_image, caption="Segmented Image with Bounding Boxes", use_column_width=True)
        st.write(f"Number of Persons Detected: {len(bboxes)}")
    else:
        st.image(color_pred, caption="Segmented Image", use_column_width=True)
        st.write("No bounding boxes available for this image.")
