import gradio as gr
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# -------------------------------
# 1️⃣ Load your trained Keras model
# -------------------------------
model = load_model("best_model.keras")  # make sure this file is in the same folder
print("Keras model loaded!")

# -------------------------------
# 2️⃣ Define your class names
# -------------------------------
class_names = sorted([
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust',
    'Apple___healthy', 'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus'
])

# -------------------------------
# 3️⃣ Define prediction function
# -------------------------------
def predict_leaf(img):
    if img is None:
        return "No image provided"
    
    # Convert to RGB if needed
    if img.mode != "RGB":
        img = img.convert("RGB")
    
    # Resize and scale
    img = img.resize((224, 224))
    x = np.array(img) / 255.0
    x = np.expand_dims(x, axis=0)  # add batch dimension
    
    # Prediction
    preds = model.predict(x)
    class_idx = np.argmax(preds, axis=1)[0]
    return class_names[class_idx]

# -------------------------------
# 4️⃣ Create Gradio interface
# -------------------------------
demo = gr.Interface(
    fn=predict_leaf,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Plant Disease Classifier",
    description="Upload a leaf image and get the predicted disease."
)

# Launch app
demo.launch()
