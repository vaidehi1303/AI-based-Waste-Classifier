import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import os

# -----------------------------
# Load the trained model
# -----------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("mobilenet_trashnet.h5")
    return model

model = load_model()

# Class labels
CLASS_NAMES = ['cardboard', 'glass', 'metal', 'paper', 'plastic']

# Log file path
LOG_FILE = "waste_log.csv"

# -----------------------------
# Function to preprocess image
# -----------------------------
def preprocess_image(image):
    image = image.resize((224, 224))  # MobileNetV2 size
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    return img_array

# -----------------------------
# Function to get recycling tips
# -----------------------------
def get_recycling_tip(label):
    tips = {
        "plastic": "🧴 Rinse containers, remove caps, and check recycling number before disposal.",
        "paper": "📄 Keep paper clean and dry. Avoid oily or food-stained paper.",
        "metal": "🥫 Clean cans, remove paper labels if possible, and crush to save space.",
        "glass": "🥂 Rinse bottles, remove lids, and sort by color if required. Avoid mixing with ceramics.",
        "cardboard": "📦 Flatten boxes, remove plastic tape, and keep them dry before recycling.",
        "trash": "🚮 Non-recyclable items should go into general waste. Consider reusing when possible."
    }
    return tips.get(label, "♻️ Dispose responsibly.")

# -----------------------------
# Function to log predictions
# -----------------------------
def log_prediction(image_name, predicted_label, confidence):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_entry = pd.DataFrame(
        [[image_name, predicted_label.upper(), f"{confidence:.2f}", timestamp]],
        columns=["Image Name", "Predicted Class", "Confidence (%)", "Timestamp"]
    )
    
    if os.path.exists(LOG_FILE):
        old_data = pd.read_csv(LOG_FILE)
        updated_data = pd.concat([old_data, new_entry], ignore_index=True)
    else:
        updated_data = new_entry
    
    updated_data.to_csv(LOG_FILE, index=False)

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Smart Waste Classifier", layout="wide", page_icon="♻️")
st.title("♻️ Smart Waste Classification & Management using AI")
st.sidebar.title("Navigation")
menu = st.sidebar.radio("Go to:", [
    "Single Image Prediction",
    "Batch Prediction",
    "Camera Prediction",
    "Waste Log",
    "Recycling Information Center",
    "About Project"
])

# -----------------------------
# SINGLE IMAGE PREDICTION
# -----------------------------
if menu == "Single Image Prediction":
    st.subheader("📸 Single Image Prediction")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        img_array = preprocess_image(image)
        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        predicted_label = CLASS_NAMES[np.argmax(score)]
        confidence = 100 * np.max(score)

        st.markdown(f"<h3 style='color:#00ffcc;'>🧠 Predicted Class: <b>{predicted_label.upper()}</b></h3>", unsafe_allow_html=True)
        st.markdown(f"<p style='font-size:18px;'>Confidence: <b>{confidence:.2f}%</b></p>", unsafe_allow_html=True)

        # Dark-themed chart
        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor('#0e1117')
        ax.set_facecolor('#0e1117')
        ax.bar(CLASS_NAMES, score.numpy(), color='#00ffcc')
        ax.set_ylabel("Confidence", color='white')
        ax.set_title("Prediction Confidence per Class", color='white')
        ax.tick_params(colors='white')
        for i, v in enumerate(score.numpy()):
            ax.text(i, v + 0.01, f"{v*100:.1f}%", ha='center', color='white', fontweight='bold')
        plt.xticks(rotation=30)
        st.pyplot(fig)

        # Recycling tip
        st.markdown(
            f"<div style='background-color:#1b2b2b; padding:12px; border-radius:12px; border-left:5px solid #00ffcc;'>"
            f"<h4 style='color:#00ffcc;'>♻️ Proper Recycling Method:</h4>"
            f"<p style='font-size:16px; color:white;'>{get_recycling_tip(predicted_label)}</p>"
            "</div>",
            unsafe_allow_html=True
        )

        # Log this prediction
        log_prediction(uploaded_file.name, predicted_label, confidence)

# -----------------------------
# BATCH PREDICTION
# -----------------------------
elif menu == "Batch Prediction":
    st.subheader("📦 Batch Waste Classification")
    uploaded_files = st.file_uploader("Upload multiple waste images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if uploaded_files:
        data = []
        for uploaded_file in uploaded_files:
            image = Image.open(uploaded_file)
            img_array = preprocess_image(image)
            predictions = model.predict(img_array)
            score = tf.nn.softmax(predictions[0])
            predicted_label = CLASS_NAMES[np.argmax(score)]
            confidence = 100 * np.max(score)
            data.append({
                "Image Name": uploaded_file.name,
                "Predicted Class": predicted_label.upper(),
                "Confidence (%)": f"{confidence:.2f}",
                "Recycling Tip": get_recycling_tip(predicted_label)
            })
            # Log this prediction
            log_prediction(uploaded_file.name, predicted_label, confidence)

        st.markdown("<h4 style='color:#00ffcc;'>Batch Prediction Summary</h4>", unsafe_allow_html=True)
        df = pd.DataFrame(data)
        st.dataframe(df.style.set_properties(**{
            'background-color': '#1b2b2b', 'color': 'white', 'border-color': '#00ffcc'
        }))

# -----------------------------
# CAMERA PREDICTION
# -----------------------------
elif menu == "Camera Prediction":
    st.subheader("📷 Real-time Camera Prediction")
    camera_image = st.camera_input("Take a photo of the waste item")

    if camera_image:
        image = Image.open(camera_image)
        st.image(image, caption="Captured Image", use_container_width=True)

        img_array = preprocess_image(image)
        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        predicted_label = CLASS_NAMES[np.argmax(score)]
        confidence = 100 * np.max(score)

        st.markdown(f"<h3 style='color:#00ffcc;'>🧠 Predicted Class: <b>{predicted_label.upper()}</b></h3>", unsafe_allow_html=True)
        st.markdown(f"<p style='font-size:18px;'>Confidence: <b>{confidence:.2f}%</b></p>", unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor('#0e1117')
        ax.set_facecolor('#0e1117')
        ax.bar(CLASS_NAMES, score.numpy(), color='#00ffcc')
        ax.set_ylabel("Confidence", color='white')
        ax.set_title("Prediction Confidence per Class", color='white')
        ax.tick_params(colors='white')
        for i, v in enumerate(score.numpy()):
            ax.text(i, v + 0.01, f"{v*100:.1f}%", ha='center', color='white', fontweight='bold')
        plt.xticks(rotation=30)
        st.pyplot(fig)

        st.markdown(
            f"<div style='background-color:#1b2b2b; padding:12px; border-radius:12px; border-left:5px solid #00ffcc;'>"
            f"<h4 style='color:#00ffcc;'>♻️ Proper Recycling Method:</h4>"
            f"<p style='font-size:16px; color:white;'>{get_recycling_tip(predicted_label)}</p>"
            "</div>",
            unsafe_allow_html=True
        )

        # Log this prediction
        log_prediction("Camera Capture", predicted_label, confidence)

# -----------------------------
# WASTE LOG MANAGEMENT
# -----------------------------
elif menu == "Waste Log":
    st.subheader("🧾 Waste Log Management System")
    if os.path.exists(LOG_FILE):
        df = pd.read_csv(LOG_FILE)
        st.dataframe(df.style.set_properties(**{
            'background-color': '#1b2b2b', 'color': 'white', 'border-color': '#00ffcc'
        }))
        st.download_button("📥 Download Log as CSV", data=open(LOG_FILE, "rb"), file_name="waste_log.csv", mime="text/csv")
    else:
        st.warning("No logs found yet. Make some predictions first!")

# -----------------------------
# RECYCLING INFORMATION CENTER
# -----------------------------
elif menu == "Recycling Information Center":
    st.subheader("📚 Recycling Information Center")
    st.write("Learn about how to properly recycle different types of materials:")

    info = {
        "Plastic": "Rinse and dry before recycling. Avoid single-use plastics and try reusing containers.",
        "Paper": "Keep paper clean and dry. Avoid recycling coated or oily paper.",
        "Glass": "Rinse bottles and jars, remove caps, and separate by color if required.",
        "Metal": "Rinse cans, remove labels, and crush to save space in bins.",
        "Cardboard": "Flatten boxes, keep dry, and remove any plastic or tape before recycling."
    }

    for material, tip in info.items():
        st.markdown(
            f"<div style='background-color:#1b2b2b; padding:12px; border-radius:12px; border-left:5px solid #00ffcc; margin-bottom:10px;'>"
            f"<h4 style='color:#00ffcc;'>{material}</h4>"
            f"<p style='color:white; font-size:16px;'>{tip}</p>"
            "</div>",
            unsafe_allow_html=True
        )

# -----------------------------
# ABOUT PROJECT
# -----------------------------
elif menu == "About Project":
    st.subheader("About the Project 🌍")
    st.write("""
    This project uses **MobileNetV2**, a pre-trained CNN, fine-tuned to classify waste into five categories:
    - Cardboard
    - Glass
    - Metal
    - Paper
    - Plastic

    **Features:**
    - Single, batch, and real-time predictions
    - Waste Log Management System (CSV export)
    - Recycling Information Center for awareness
    - Streamlit-based dark-themed interactive UI

    **Future Scope:**
    - Smart bin integration
    - Cloud-based analytics dashboard
    - Automated waste sorting simulation
    """)

   
