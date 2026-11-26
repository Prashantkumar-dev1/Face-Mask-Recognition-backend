import io
from pathlib import Path
from flask import Flask, jsonify, request
from PIL import Image, UnidentifiedImageError
from flask_cors import CORS

from model_engine import MaskClassifier

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best_mask_model.h5"
if not MODEL_PATH.exists():
    MODEL_PATH = BASE_DIR / "mask_classifier_model.h5"

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB max upload

CORS(
    app,
    resources={r"/*": {
        "origins": [
            "https://face-mask-recognition-frontend.vercel.app"
        ]
    }},
    methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"]
)
classifier = MaskClassifier(MODEL_PATH)


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    return response


@app.route("/", methods=["GET"])
@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "ok",
            "model_loaded": True,
            "model": MODEL_PATH.name,
        }
    )


@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        return ("", 204)

    uploaded = request.files.get("file") or request.files.get("image")
    if uploaded is None or not uploaded.filename:
        return jsonify({"error": "No image file provided. Send multipart form field 'file'."}), 400

    raw_bytes = uploaded.read()
    if not raw_bytes:
        return jsonify({"error": "Uploaded file is empty."}), 400

    try:
        image = Image.open(io.BytesIO(raw_bytes))
        image.load()
    except (UnidentifiedImageError, OSError):
        return jsonify({"error": "Invalid or unsupported image file."}), 400

    result = classifier.predict_image(image)
    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
