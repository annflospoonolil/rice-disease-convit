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
language = st.selectbox(
    "🌐 Select Language",
    ["English", "Malayalam"]
)
if language == "Malayalam":
    TEXTS = {
        "upload_title": "📷 ചിത്രം അപ്ലോഡ് ചെയ്യുക അല്ലെങ്കിൽ പകർത്തുക",
        "choose_method": "ഇൻപുട്ട് രീതി തിരഞ്ഞെടുക്കുക:",
        "upload": "ചിത്രം അപ്ലോഡ് ചെയ്യുക",
        "camera": "ഫോട്ടോ എടുക്കുക",
        "prediction": "ഫലം",
        "confidence": "വിശ്വാസ്യത",
        "top_predictions": "മുൻനിര പ്രവചനങ്ങൾ",
        "suggested_action": "പരിഹാര നിർദ്ദേശങ്ങൾ",
        "healthy_msg": "വിള ആരോഗ്യകരമാണ്.",
        "disease_msg": "രോഗം കണ്ടെത്തി. ദയവായി നിർദ്ദേശങ്ങൾ പിന്തുടരുക.",
        "gradcam": "🔍 മോഡൽ ശ്രദ്ധിച്ച ഭാഗങ്ങൾ",
        "explanation": "വിവരണം"
    }
else:
    TEXTS = {
        "upload_title": "📷 Upload or Capture Image",
        "choose_method": "Choose input method:",
        "upload": "Upload Image",
        "camera": "Take Photo",
        "prediction": "Prediction Result",
        "confidence": "Confidence",
        "top_predictions": "Top 3 Predictions",
        "suggested_action": "Suggested Action",
        "healthy_msg": "Your crop looks healthy.",
        "disease_msg": "Disease detected. Follow the tips below.",
        "gradcam": "🔍 Model Attention (Grad-CAM)",
        "explanation": "Explanation"
    }
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
    if language == "Malayalam":
        explanations = {
            "Rice Blast": "മോഡൽ റൈസ് ബ്ലാസ്റ്റ് രോഗത്തിൽ സാധാരണ കാണപ്പെടുന്ന ഗ്രേ സെന്ററുകളും ഡാർക്ക് ബോർഡറുകളും ഉള്ള സ്പിൻഡിൽ ആകൃതിയിലുള്ള ലെഷനുകൾ കണ്ടെത്തി.",
            
            "Brown Spot": "മോഡൽ ബ്രൗൺ സ്പോട്ട് രോഗത്തിന്റെ സാധാരണ ലക്ഷണമായ പത്രത്തിൽ നിരവധി ബ്രൗൺ വൃത്താകൃതിയിലുള്ള സ്പോട്ടുകൾ തിരിച്ചറിയുന്നു.",
            
            "Healthy": "ഇല പച്ച നിറത്തിൽ ഒരേപോലെ നിലകൊള്ളുന്നു, രോഗത്തിന്റെ ദൃശ്യമായ പാറ്റേണുകൾ ഇല്ല, ഇത് ആരോഗ്യകരമായ വിളയാണെന്ന് സൂചിപ്പിക്കുന്നു.",
            
            "Rice Hispa": "മോഡൽ റൈസ് ഹിസ്പാ അറ്റാക്കിന്റെ പ്രത്യേകതയായ scraped അല്ലെങ്കിൽ damaged leaf surfaces കണ്ടെത്തി.",
            
            "Rice Scald": "മോഡൽ വെള്ളം നനഞ്ഞതുപോലെയുള്ള പാടുകൾ കണ്ടെത്തി, ഇത് റൈസ് സ്കാൾഡ് രോഗത്തിന് സാധാരണ ലക്ഷണമാണ്."
        }
    else:
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
    if language == "Malayalam":
        tips = {
            "Rice Blast": [
                "ശരിയായ ഫീൽഡ് ഡ്രൈനേജ് നിലനിർത്തുക",
                "അധിക നൈട്രജൻ വളം ഒഴിവാക്കുക",
                "ആവശ്യമായപ്പോൾ ശുപാർശ ചെയ്ത ഫംഗിസൈഡുകൾ പ്രയോഗിക്കുക",
                "പ്രതിരോധ ശേഷിയുള്ള റൈസ് വർഗ്ഗങ്ങൾ ഉപയോഗിക്കുക"
            ],
            
            "Brown Spot": [
                "സന്തുലിതമായ മണ്ണ് പോഷണം ഉറപ്പാക്കുക",
                "പൊട്ടാസ്യം വളങ്ങൾ പ്രയോഗിക്കുക",
                "വിളയിൽ ജലസമ്മർദ്ദം ഒഴിവാക്കുക",
                "രോഗമുക്ത വിത്തുകൾ ഉപയോഗിക്കുക"
            ],
            
            "Healthy": [
                "നിയമിതമായ നിരീക്ഷണം തുടരുക",
                "ശരിയായ ജലസേചനം നിലനിർത്തുക",
                "സന്തുലിതമായ വളങ്ങൾ ഉപയോഗിക്കുക",
                "പുല്ല്‍ ഇല്ലാത്ത ഫീൽഡ് സൂക്ഷിക്കുക"
            ],
            
            "Rice Hispa": [
                "ബാധിത ഇലകൾ നീക്കം ചെയ്ത് നശിപ്പിക്കുക",
                "ശുപാർശ ചെയ്ത കീടനാശിനികൾ ഉപയോഗിക്കുക",
                "സ്വാഭാവിക ശത്രുക്കളെ പ്രോത്സാഹിപ്പിക്കുക",
                "തോട്ടത്തിലെ സസ്യങ്ങൾ ഒതുക്കി നിർത്തുന്നത് ഒഴിവാക്കുക"
            ],
            
            "Rice Scald": [
                "അധിക ജല നിശ്ചലത ഒഴിവാക്കുക",
                "ആവശ്യമായപ്പോൾ ഫംഗിസൈഡുകൾ പ്രയോഗിക്കുക",
                "സസ്യങ്ങൾ തമ്മിൽ ശരിയായ ഇടവേള ഉറപ്പാക്കുക",
                "പ്രതിരോധ ശേഷിയുള്ള വർഗ്ഗങ്ങൾ ഉപയോഗിക്കുക"
            ]
        }
    else:
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

st.subheader(TEXTS["upload_title"])

input_option = st.radio(
    TEXTS["choose_method"],
    [TEXTS["upload"], TEXTS["camera"]]
)

image = None

if input_option == TEXTS["upload"]:
    uploaded_file = st.file_uploader(
        TEXTS["upload_title"],
        type=["jpg", "jpeg", "png"]
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

elif input_option == TEXTS["camera"]:
    camera_image = st.camera_input(TEXTS["camera"])

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
    st.session_state.prediction = predicted_label
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
    st.subheader(TEXTS["gradcam"])

    st.image(
        visualization,
        caption="Highlighted regions the model focused on",
        use_column_width=True
    )

    
    st.subheader(TEXTS["explanation"])
    st.info(final_explanation)
    st.divider()
    # ---------------------------------------------------
    # TOP 3 PREDICTIONS
    # ---------------------------------------------------
    st.subheader(TEXTS["top_predictions"])

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
    st.subheader(TEXTS["suggested_action"])

    if predicted_label == "Healthy":
        st.success(TEXTS["healthy_msg"])
    else:
        st.warning(f"{predicted_label} detected. {TEXTS['disease_msg']}")
        
    for tip in prevention_tips:
        st.markdown(f"✅ {tip}")
import google.generativeai as genai

# 🔑 Configure API
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
chat_model = genai.GenerativeModel("models/gemini-2.5-flash")
if "messages" not in st.session_state:
    st.session_state.messages = []
st.markdown("## 🌾 AI Farming Assistant")

# 🧠 Store chat history
history = ""

for msg in st.session_state.messages:
    role = "User" if msg["role"] == "user" else "Assistant"
    history += f"{role}: {msg['content']}\n"
# 🧠 Store chat history

# 🧾 Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 💬 Chat input (THIS IS STREAMLIT TEMPLATE 🔥)
user_input = st.chat_input("Ask anything about crops, diseases, fertilizers...")

if user_input:
    if language == "Malayalam":
        lang_instruction = "Answer in Malayalam (simple, farmer-friendly language)."
    else:
        lang_instruction = "Answer in simple English."
    # Show user message
    with st.chat_message("user"):
        st.markdown(user_input)

    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    if "prediction" in st.session_state:
        prompt = f"""
        You are an agriculture expert in rice leaf diseases.

        {lang_instruction}

        Detected disease: {st.session_state.prediction}

        Conversation so far:
        {history}

        User: {user_input}

        Continue the conversation naturally and give helpful advice.
        """
    else:
        prompt = f"""
        You are an agriculture expert in rice leaf diseases.

        {lang_instruction}

        Conversation so far:
        {history}

        User: {user_input}

        Continue the conversation naturally.
        """
    response = chat_model.generate_content(prompt)
    bot_reply = response.text

    # Show bot response
    with st.chat_message("assistant"):
        st.markdown(bot_reply)

    st.session_state.messages.append({
        "role": "assistant",
        "content": bot_reply
    })