
import os
import requests
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Fruit & Vegetable Classifier", page_icon="🥦", layout="centered")
st.title("🍎 Fruit & Vegetable Classifier")
st.write("Upload an image and let your CNN predict which fruit or vegetable it contains.")

# Local default. Set API_URL in your environment when deploying elsewhere.
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)

    if st.button("Classify image", type="primary", use_container_width=True):
        with st.spinner("Sending image to FastAPI…"):
            try:
                response = requests.post(
                    f"{API_URL}/predict",
                    files={"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type or "image/jpeg")},
                    timeout=120,
                )
                response.raise_for_status()
                result = response.json()
                st.success(f"Prediction: {result['predicted_class'].title()}")
                st.metric("Confidence", f"{result['confidence']:.2f}%")
                st.subheader("Top predictions")
                for item in result["top_predictions"]:
                    st.write(f"**{item['class'].title()}** — {item['confidence']:.2f}%")
                    st.progress(min(max(float(item["confidence"]) / 100, 0.0), 1.0))
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to FastAPI. Start the backend first and check API_URL.")
            except requests.exceptions.Timeout:
                st.error("The API request timed out. Try again or check that the backend is running.")
            except requests.exceptions.HTTPError as e:
                detail = response.text if "response" in locals() else str(e)
                st.error(f"FastAPI returned an error: {detail}")
            except requests.exceptions.RequestException as e:
                st.error(f"Request failed: {e}")

with st.expander("Connection details"):
    st.write(f"FastAPI URL: `{API_URL}`")
    st.write("API documentation: open the backend URL followed by `/docs`.")
