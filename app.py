import os
import io
import numpy as np
from flask import Flask, request, render_template_string
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from PIL import Image

app = Flask(__name__)

# ================================
# 1️⃣ TRAIN OR LOAD THE CNN MODEL
# ================================
MODEL_PATH = "mnist_cnn_model.h5"

def create_and_train_model():
    print("🔄 Training model... This will take a few minutes on first run.")
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalize & reshape
    x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
        MaxPooling2D((2,2)),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D((2,2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=3, batch_size=128, validation_data=(x_test, y_test))
    model.save(MODEL_PATH)
    print(f"✅ Model trained and saved as {MODEL_PATH}")
    return model

# Load model if available
if os.path.exists(MODEL_PATH):
    print("✅ Found existing model. Loading it...")
    model = load_model(MODEL_PATH)
else:
    model = create_and_train_model()

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
