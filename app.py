import os
import io
import numpy as np
from flask import Flask, request, render_template_string
from tensorflow.keras.models import load_model
from PIL import Image
import requests

app = Flask(__name__)

# ================================
# 1️⃣ LOAD PRE-TRAINED CNN MODEL
# ================================
MODEL_PATH = "mnist_cnn_model.h5"
MODEL_URL = "https://github.com/ghoshrajarshi425/Handwritten-Digit-Recognition/releases/download/v1.0/mnist_cnn_model.h5"

def download_model():
    """Download the pre-trained model from GitHub release."""
    print("⬇️ Downloading pre-trained model...")
    response = requests.get(MODEL_URL, stream=True)
    if response.status_code == 200:
        with open(MODEL_PATH, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("✅ Model downloaded successfully")
        return True
    return False

# Load or download model
if os.path.exists(MODEL_PATH):
    print("✅ Found existing model. Loading it...")
    model = load_model(MODEL_PATH)
else:
    if download_model():
        model = load_model(MODEL_PATH)
    else:
        raise RuntimeError("❌ Could not download pre-trained model. Please check the MODEL_URL.")

# ================================
# 2️⃣ SIMPLE HTML PAGE
# ================================
HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>Handwritten Digit Recognition</title>
    <style>
        body { font-family: Arial; text-align: center; background: #f2f2f2; margin-top: 40px; }
        form { background: white; padding: 20px; border-radius: 10px; display: inline-block; box-shadow: 0 0 10px gray; }
        input[type=file] { margin: 10px; }
        h1 { color: #333; }
    </style>
</head>
<body>
    <h1>🖋️ Handwritten Digit Recognition (0-9)</h1>
    <form action="/predict" method="post" enctype="multipart/form-data">
        <input type="file" name="file" accept="image/*" required><br>
        <input type="submit" value="Predict Digit">
    </form>
    {% if result is not none %}
        <h2>Predicted Digit: <b>{{ result }}</b></h2>
    {% endif %}
</body>
</html>
"""

# ================================
# 3️⃣ FLASK ROUTES
# ================================
@app.route("/", methods=["GET"])
def index():
    return render_template_string(HTML_PAGE, result=None)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        file = request.files.get("file")
        if not file:
            return render_template_string(HTML_PAGE, result="No file uploaded")

        # Read image and convert to grayscale
        image = Image.open(io.BytesIO(file.read())).convert("L")

        # Pillow compatibility for resampling attribute
        try:
            resample = Image.Resampling.LANCZOS
        except AttributeError:
            resample = Image.LANCZOS

        image = image.resize((28, 28), resample)

        img_array = np.array(image)
        img_array = img_array.reshape(1, 28, 28, 1).astype("float32") / 255.0

        prediction = int(np.argmax(model.predict(img_array, verbose=0)))
        return render_template_string(HTML_PAGE, result=prediction)
    except Exception as e:
        return render_template_string(HTML_PAGE, result=f"Error: {e}")


# ================================
# 4️⃣ RUN APP
# ================================
if __name__ == "__main__":
    # Use PORT environment variable if provided (useful for platforms like Heroku)
    port = int(os.environ.get("PORT", 5000))
    # Bind to 0.0.0.0 so it is reachable from other machines/containers
    app.run(host="0.0.0.0", port=port, debug=True)
