import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import pandas as pd
from model import get_model
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import cv2
import numpy as np
def reshape_transform(tensor, height=14, width=14):
    tensor = tensor[:, 1:, :]  # remove class token
    result = tensor.reshape(tensor.size(0), height, width, tensor.size(2))
    result = result.permute(0, 3, 1, 2)
    return result


st.set_page_config(
    page_title="Rice Disease Detector",
    page_icon="🌾",
    layout="centered"
)


st.markdown("""
    <style>
    .main {
        background-color: #f4f8f4;
    }
    .stButton>button {
        background-color: #2e7d32;
        color: white;
        border-radius: 10px;
        height: 3em;
        width: 100%;
        font-size: 16px;
    }
    .result-box {
        padding: 20px;
        border-radius: 15px;
        background-color: #ffffff;
        box-shadow: 0px 4px 15px rgba(0,0,0,0.1);
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)


st.markdown("<h1 style='text-align: center;'>🌾 Rice Disease Detection System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>AI-powered ConViT Model for Farmers</p>", unsafe_allow_html=True)
st.divider()


CLASS_NAMES = [
    "Rice Blast",
    "Brown Spot",
    "Healthy",
    "Rice Hispa",
    "Rice Scald"
]

NUM_CLASSES = len(CLASS_NAMES)


@st.cache_resource
def load_model():
    model = get_model(num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load("best_model.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()
target_layers = [model.blocks[-1].norm1]
cam = GradCAM(
    model=model,
    target_layers=target_layers,
    reshape_transform=reshape_transform
)

def transform_image(image):
    import cv2
    import numpy as np

    # Convert PIL → numpy
    img = np.array(image)

    # Convert RGB → LAB
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

    # Split channels
    l, a, b = cv2.split(lab)

    # Apply CLAHE on L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)

    # Merge back
    lab = cv2.merge((l, a, b))

    # Convert back to RGB
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    # Back to PIL
    image = Image.fromarray(img)

    # ---- Existing transforms ----
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return transform(image).unsqueeze(0)

def generate_explanation(label, confidence):
    explanations = {
        "Rice Blast": "The model detected spindle-shaped lesions with gray centers and dark borders, commonly seen in Rice Blast disease.",
        
        "Brown Spot": "The model identified multiple brown circular spots on the leaf, which is a typical symptom of Brown Spot disease.",
        
        "Healthy": "The leaf appears uniformly green with no visible disease patterns, indicating a healthy crop.",
        
        "Rice Hispa": "The model detected scraped or damaged leaf surfaces, which is characteristic of Rice Hispa infestation.",
        
        "Rice Scald": "The model identified irregular lesions with a water-soaked appearance, typical of Rice Scald disease."
    }

    base = explanations.get(label, "The model detected patterns based on learned features.")
    return f"{base} (Confidence: {confidence:.2f}%)"


def analyze_cam(grayscale_cam):
    import numpy as np

    max_pos = np.unravel_index(np.argmax(grayscale_cam), grayscale_cam.shape)
    h, w = grayscale_cam.shape

    if max_pos[0] < h/3:
        vertical = "upper"
    elif max_pos[0] < 2*h/3:
        vertical = "middle"
    else:
        vertical = "lower"

    if max_pos[1] < w/3:
        horizontal = "left"
    elif max_pos[1] < 2*w/3:
        horizontal = "center"
    else:
        horizontal = "right"

    return f"{vertical}-{horizontal} region"

def get_prevention_tips(label):
    tips = {
        "Rice Blast": [
            "Maintain proper field drainage",
            "Avoid excessive nitrogen fertilizer",
            "Apply recommended fungicides if needed",
            "Use resistant rice varieties"
        ],
        
        "Brown Spot": [
            "Ensure balanced soil nutrition",
            "Apply potassium fertilizers",
            "Avoid water stress in crops",
            "Use disease-free seeds"
        ],
        
        "Healthy": [
            "Continue regular monitoring",
            "Maintain proper irrigation",
            "Use balanced fertilizers",
            "Keep field free from weeds"
        ],
        
        "Rice Hispa": [
            "Remove and destroy affected leaves",
            "Use recommended insecticides",
            "Encourage natural predators",
            "Avoid overcrowding of plants"
        ],
        
        "Rice Scald": [
            "Avoid excessive water stagnation",
            "Apply fungicides if required",
            "Ensure proper spacing between plants",
            "Use resistant varieties"
        ]
    }

    return tips.get(label, ["Follow general crop management practices"])

st.subheader("📷 Upload or Capture Image")

input_option = st.radio(
    "Choose input method:",
    ["Upload Image", "Take Photo"]
)

image = None

if input_option == "Upload Image":
    uploaded_file = st.file_uploader(
        "Upload Rice Leaf Image",
        type=["jpg", "jpeg", "png"]
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

elif input_option == "Take Photo":
    camera_image = st.camera_input("Take a photo of the leaf")

    if camera_image is not None:
        image = Image.open(camera_image).convert("RGB")


if image is not None:

    col1, col2 = st.columns([1, 1])

    # image = Image.open(uploaded_file).convert("RGB")

    with col1:
        st.image(image, caption="Input Image", use_column_width=True)

    input_tensor = transform_image(image)
    model.zero_grad()

    outputs = model(input_tensor)
    probabilities = F.softmax(outputs, dim=1)[0]

    confidence, predicted_class = torch.max(probabilities, 0)

    targets = [ClassifierOutputTarget(predicted_class.item())]

    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0]
    predicted_label = CLASS_NAMES[predicted_class.item()]
    confidence_score = confidence.item() * 100
    prevention_tips = get_prevention_tips(predicted_label)
    # ---- TEXT EXPLANATION ----
    region = analyze_cam(grayscale_cam)
    explanation_text = generate_explanation(predicted_label, confidence_score)

    final_explanation = f"{explanation_text} The model mainly focused on the {region} of the leaf."
    
    rgb_img = np.array(image.resize((224, 224))).astype(np.float32) / 255.0

    visualization = show_cam_on_image(
        rgb_img,
        grayscale_cam,
        use_rgb=True
    )


    with col2:
        st.markdown("###  Prediction Result")
        
        st.markdown(f"""
            <div class="result-box">
                <h2 style='color:#2e7d32;'>{predicted_label}</h2>
                <h4>Confidence: {confidence_score:.2f}%</h4>
            </div>
        """, unsafe_allow_html=True)

        st.progress(int(confidence_score))

    st.divider()
    st.subheader("🔍 Model Attention (Grad-CAM)")

    st.image(
        visualization,
        caption="Highlighted regions the model focused on",
        use_column_width=True
    )

    
    st.subheader("Explanation")

    st.info(final_explanation)
    st.divider()
    # ---------------------------------------------------
    # TOP 3 PREDICTIONS
    # ---------------------------------------------------
    st.subheader("📊 Top 3 Predictions")

    probs_df = pd.DataFrame({
        "Disease": CLASS_NAMES,
        "Probability (%)": probabilities.detach().cpu().numpy() * 100
    })

    probs_df = probs_df.sort_values(by="Probability (%)", ascending=False)

    st.bar_chart(probs_df.set_index("Disease"))

    st.dataframe(probs_df.head(3), use_container_width=True)

    # ---------------------------------------------------
    # HEALTH ADVICE SECTION
    # ---------------------------------------------------
    st.subheader("🌱 Suggested Action")

    if predicted_label == "Healthy":
        st.success("Your crop looks healthy. Keep monitoring regularly.")
    else:
        st.warning(f"{predicted_label} detected. Ensure following tips are followed and consider consulting an agricultural expert or applying appropriate treatment.")
        
    for tip in prevention_tips:
        st.markdown(f"✅ {tip}")