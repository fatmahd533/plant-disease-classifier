import gradio as gr
import tensorflow as tf
from PIL import Image
import numpy as np

# Load model
model = tf.keras.models.load_model("best_model.keras")
class_names = [
    # Replace with your actual class names
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    # ... all 38 classes
]

# Prediction function
def predict_image(img):
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    preds = model.predict(img_array)
    class_idx = np.argmax(preds)
    return class_names[class_idx]

# Gradio interface
demo = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Plant Disease Classifier",
    description="Upload a leaf image and get the predicted disease."
)

demo.launch()
