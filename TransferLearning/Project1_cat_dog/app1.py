import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from PIL import Image
import os

# Page config
st.set_page_config(page_title="🐾 Cat vs Dog Classifier", layout="centered")

# Constants
MODEL_PATH = 'models/cat_dog_transfer_VGG16_model.h5'
CLASS_LABELS = ['Cat', 'Dog']

# Load model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()


# Header
st.markdown(
    """
    <div style='text-align: center;'>
        <h1 style='color:#4B8BBE;'>🐾 Cat vs Dog Classifier using TransferLearning</h1>
        <p style='font-size: 18px;'>Upload an image to find out whether it's a cat or a dog using a ResNet50 deep learning model.</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Upload
uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and display image
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width =True)

    # Preprocess
    img_resized = img.resize((224, 224))
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_preprocessed = preprocess_input(img_array)

    # Predict
    with st.spinner("Making prediction..."):
        prediction = model.predict(img_preprocessed)[0][0]
        predicted_class = CLASS_LABELS[int(prediction > 0.5)]
        confidence = float(prediction if predicted_class == 'Dog' else 1 - prediction)

    # Result Card
    st.markdown("---")
    st.markdown(
        f"""
        <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; text-align: center;">
            <h2 style="color: #333;">Prediction: <span style="color: #4B8BBE;">{predicted_class}</span></h2>
            <p style="font-size: 18px;">Model confidence: <strong>{confidence:.2%}</strong></p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Confidence Bar
    st.progress(confidence)

    # Info
    st.info("👆 Closer to 100% means the model is more confident in its prediction.")
else:
    st.markdown("📎 Supported formats: JPG, JPEG, PNG")
