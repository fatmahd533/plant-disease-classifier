import gradio as gr
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from model import PlantCNN

# -------------------------------
# 1️⃣ Classes du modèle
# -------------------------------
class_names = sorted(['Tomato___Late_blight', 'Tomato___healthy', 'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Potato___healthy', 'Corn_(maize)___Northern_Leaf_Blight', 'Tomato___Early_blight', 'Tomato___Septoria_leaf_spot', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Strawberry___Leaf_scorch', 'Peach___healthy', 'Apple___Apple_scab', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Bacterial_spot', 'Apple___Black_rot', 'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Peach___Bacterial_spot', 'Apple___Cedar_apple_rust', 'Tomato___Target_Spot', 'Pepper,_bell___healthy', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Potato___Late_blight', 'Tomato___Tomato_mosaic_virus', 'Strawberry___healthy', 'Apple___healthy', 'Grape___Black_rot', 'Potato___Early_blight', 'Cherry_(including_sour)___healthy', 'Corn_(maize)___Common_rust_', 'Grape___Esca_(Black_Measles)', 'Raspberry___healthy', 'Tomato___Leaf_Mold', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Pepper,_bell___Bacterial_spot', 'Corn_(maize)___healthy'])
num_classes = len(class_names)  # ✅ 38 classes

device = torch.device("cpu")

# -------------------------------
# 2️⃣ Charger le modèle
# -------------------------------
model = PlantCNN(num_classes)
model.load_state_dict(torch.load("best_model_finetuned.pth", map_location=device))
model.eval()

# -------------------------------
# 3️⃣ Transformations pour l'image
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# -------------------------------
# 4️⃣ Fonction de prédiction
# -------------------------------
def predict_leaf(img: Image.Image):
    if img.mode != "RGB":
        img = img.convert("RGB")
    x = transform(img).unsqueeze(0)
    with torch.no_grad():
        outputs = model(x)
        probs = F.softmax(outputs, dim=1)
        class_idx = torch.argmax(probs, dim=1).item()
    return class_names[class_idx]

# -------------------------------
# 5️⃣ Interface Gradio
# -------------------------------
demo = gr.Interface(
    fn=predict_leaf,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Plant Disease Classifier (PyTorch)",
    description="Upload a leaf image and get the predicted disease."
)

demo.launch()