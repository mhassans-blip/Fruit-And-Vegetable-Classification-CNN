# Fruit & Vegetable Classification — FastAPI + Streamlit

This project wraps the trained CNN from your notebook in a FastAPI backend and a Streamlit frontend.

## 1. Add your trained model

Copy these two files into this project folder:

- `fruit_vegetable_custom_cnn_final.keras`
- `fruit_vegetable_class_names.json` (included here with the 36 class names shown in your notebook)

The `.keras` model file is not included because the uploaded notebook does not contain the actual trained model weights as a separate file. Export/download it from the environment where you trained it.

In Colab, after running the notebook's save cell, download the model:

```python
from google.colab import files
files.download('/content/fruit_vegetable_custom_cnn_final.keras')
files.download('/content/fruit_vegetable_class_names.json')
```

If your model is saved in Google Drive, update the path accordingly before downloading.

## 2. Create and activate a virtual environment (Windows PowerShell)

Open PowerShell in this folder:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, use Command Prompt instead:

```bat
.venv\Scripts\activate.bat
```

## 3. Run FastAPI (terminal 1)

Make sure the `.keras` file and JSON file are in this folder.

```powershell
uvicorn main:app --reload
```

Test:
- API health: http://127.0.0.1:8000/health
- Interactive API docs: http://127.0.0.1:8000/docs

The prediction endpoint is `POST /predict`; it expects an uploaded image in the multipart field named `file`.

## 4. Run Streamlit (terminal 2)

Keep FastAPI running. Open a second terminal in the same folder, activate the same environment, and run:

```powershell
streamlit run streamlit_app.py
```

Streamlit opens a local browser page, usually at http://localhost:8501. Upload an image and click **Classify image**. Streamlit sends the image to FastAPI, and displays the returned prediction and top-five scores.

## How the connection works

`streamlit_app.py` sends an HTTP POST request to `http://127.0.0.1:8000/predict`. FastAPI loads the model once at startup, preprocesses the uploaded image to RGB 224x224, and returns JSON containing the predicted class, confidence, and top-five predictions.

## Deploying publicly

For a public deployment, host the FastAPI backend and Streamlit frontend as separate services. Set the Streamlit service's `API_URL` environment variable to the public base URL of the FastAPI service, for example `https://your-api.example.com` (no trailing `/`). Do not leave it at `127.0.0.1` in a hosted frontend: that would point to the frontend container itself, not the backend.

Keep the model file available to the backend service. Use HTTPS, set upload-size limits, and avoid committing secrets. The backend needs enough RAM for TensorFlow and the model. Exact hosting steps depend on your selected platform.

## Troubleshooting

- **Model not found:** put `fruit_vegetable_custom_cnn_final.keras` beside `main.py`, or set `MODEL_PATH`.
- **Class names file not found:** put `fruit_vegetable_class_names.json` beside `main.py`, or set `CLASS_NAMES_PATH`.
- **Streamlit cannot connect:** ensure FastAPI is running and `API_URL` points to its reachable base URL.
- **TensorFlow installation fails:** use a supported Python version for your TensorFlow release; Python 3.10 or 3.11 is a practical starting point, then install dependencies in a fresh virtual environment.
- **Wrong predictions:** ensure class order in JSON exactly matches the class order used for training. The included list is copied from the notebook output.
