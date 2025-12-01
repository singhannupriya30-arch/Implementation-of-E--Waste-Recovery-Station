from flask import Flask, request, jsonify, render_template, redirect, url_for, session
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
app.secret_key = "supersecret"   # needed for session

MODEL_PATH = "ewaste_model.h5"
model = load_model(MODEL_PATH, compile=False)

CLASSES = ['Battery','Copper Coil','Keyboard','LED','Mobile','Mouse',
           'Non-E-Waste','PCB','Television','Transistor','Wires']

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

PREDICT_DIR = "predict"
os.makedirs(PREDICT_DIR, exist_ok=True)

@app.route("/")
def home():
    result = session.pop("result", None)  # get result from session once
    return render_template("index.html", result=result)

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)[0]
    confidence = float(np.max(pred))
    pred_class = int(np.argmax(pred))

    if confidence < 0.6:
        return {"label": "Unknown / Non-E-Waste", "confidence": confidence}
    else:
        return {"label": CLASSES[pred_class], "confidence": confidence}

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    existing_files = [f for f in os.listdir(UPLOAD_DIR) if f.endswith(".jpg")]
    next_index = len(existing_files) + 1
    filename = f"image_{next_index:04d}.jpg"
    filepath = os.path.join(UPLOAD_DIR, filename)
    file.save(filepath)

    result = predict_image(filepath)
    result["saved_as"] = filename

    # ESP32 → JSON
    if request.accept_mimetypes.accept_json or request.is_json:
        return jsonify(result)

    # Browser → store in session, redirect back to "/"
    session["result"] = result
    return redirect(url_for("/"))

@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    # Always overwrite same file
    filename = "latest.jpg"
    filepath = os.path.join(PREDICT_DIR, filename)
    file.save(filepath)

    result = predict_image(filepath)
    result["saved_as"] = filename

    return jsonify(result)   # Always JSON



if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
