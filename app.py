
import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, UnidentifiedImageError

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "fruit_vegetable_custom_cnn_final.keras"
CLASS_NAMES_PATH = BASE_DIR / "fruit_vegetable_class_names.json"

st.set_page_config(
    page_title="Fruit & Vegetable Classifier",
    page_icon="🥦",
    layout="centered",
)

st.title("🍎 Fruit & Vegetable Classifier")
st.write("Upload an image to predict its fruit or vegetable class.")

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
        class_names = json.load(f)

    if model.output_shape[-1] != len(class_names):
        raise ValueError("Model output and class names do not match.")

    return model, class_names

try:
    model, class_names = load_model()
except Exception as e:
    st.error(f"Could not load model or class names: {e}")
    st.stop()

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png", "webp"],
)

if uploaded_file:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded image", use_container_width=True)

        if st.button("Classify Image", type="primary"):
            resized = image.resize((224, 224))
            array = np.asarray(resized, dtype=np.float32)
            batch = np.expand_dims(array, axis=0)

            # The model's Rescaling layer handles 0–1 normalization.
            predictions = model.predict(batch, verbose=0)[0]

            top_indices = np.argsort(predictions)[::-1][:5]
            best_index = int(top_indices[0])

            st.success(
                f"Prediction: {class_names[best_index].title()}"
            )
            st.metric(
                "Confidence",
                f"{float(predictions[best_index]) * 100:.2f}%"
            )

            st.subheader("Top 5 Predictions")
            for i in top_indices:
                confidence = float(predictions[i]) * 100
                st.write(
                    f"**{class_names[int(i)].title()}** — "
                    f"{confidence:.2f}%"
                )
                st.progress(min(max(confidence / 100, 0), 1))

    except (UnidentifiedImageError, OSError):
        st.error("Invalid image. Please upload a valid JPG or PNG.")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
