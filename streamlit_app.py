"""
Plant Disease Detector - Streamlit version
Needs plant_model.keras and classes.json in the same folder.
"""
import json
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

st.set_page_config(page_title="Plant Disease Detector", page_icon="🌿")


@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("plant_model.keras", compile=False)
    with open("classes.json") as f:
        classes = json.load(f)
    return model, classes


def pretty(name):
    return name.replace("___", " - ").replace("_", " ")


def predict(model, classes, img):
    img = img.convert("RGB").resize((224, 224))
    arr = np.expand_dims(np.array(img, dtype="float32"), 0)  # scaling is inside the model
    probs = model.predict(arr, verbose=0)[0]
    top = np.argsort(probs)[::-1][:3]
    return [(pretty(classes[i]), float(probs[i])) for i in top]


st.title("🌿 Plant Disease Detector")
st.write("Upload a clear photo of a single leaf (plain background works best). "
         "The model recognises 38 classes from the PlantVillage dataset.")

model, classes = load_model()

tab_up, tab_cam = st.tabs(["Upload image", "Use camera"])
with tab_up:
    file = st.file_uploader("Leaf image", type=["jpg", "jpeg", "png"])
with tab_cam:
    cam = st.camera_input("Take a photo of a leaf")

source = file or cam
if source is not None:
    image = Image.open(source)
    st.image(image, caption="Your image", use_container_width=True)
    with st.spinner("Analysing..."):
        results = predict(model, classes, image)
    best, conf = results[0]
    st.subheader(f"Prediction: {best}")
    st.write(f"Confidence: {conf * 100:.1f}%")
    st.write("Top 3:")
    for name, p in results:
        st.progress(min(max(p, 0.0), 1.0), text=f"{name}: {p * 100:.1f}%")

st.caption("Mini project | Transfer learning with MobileNetV2 | Test accuracy 95.6%")
