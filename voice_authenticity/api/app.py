import os
import base64
import tempfile

from flask import Flask, request, jsonify
import librosa

from voice_authenticity.voice_features.extract_features import extract_features
from voice_authenticity.models.voice_classifier import VoiceClassifier
from voice_authenticity.explain.explanation import generate_explanation


# =====================================================
# CONFIG
# =====================================================

API_KEY = "hackathon-demo-key-123"  # CHANGE before submission

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "voice_model.pkl")


# =====================================================
# APP SETUP
# =====================================================

app = Flask(__name__)
classifier = VoiceClassifier.load(MODEL_PATH)


# =====================================================
# API KEY CHECK
# =====================================================

def check_api_key(req):
    return req.headers.get("X-API-KEY") == API_KEY


# =====================================================
# HEALTH CHECK
# =====================================================

@app.route("/", methods=["GET"])
def health():
    return jsonify({
        "status": "running",
        "service": "AI Voice Authenticity Detection API"
    })


# =====================================================
# CORE DETECTION ENDPOINT
# =====================================================

@app.route("/detect", methods=["POST"])
def detect():

    if not check_api_key(request):
        return jsonify({"error": "Invalid or missing API key"}), 401

    # -----------------------------
    # GUVI sends FORM DATA, not JSON
    # -----------------------------
    data = request.get_json(silent=True)
    if data is None:
        data = request.form

    audio_b64 = (
        data.get("audio_base64_format")
        or data.get("audio_base64")
    )

    if not audio_b64:
        return jsonify({"error": "audio_base64 missing"}), 400

    try:
        audio_bytes = base64.b64decode(audio_b64)

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        y, sr = librosa.load(tmp_path, sr=16000, mono=True)
        os.remove(tmp_path)

        features = extract_features(audio_array=y, sample_rate=sr)
        prediction = classifier.predict(features)
        explanation = generate_explanation(features)

        return jsonify({
            "classification": prediction["classification"],
            "confidence": round(prediction["confidence"], 3),
            "explanation": explanation
        })

    except Exception as e:
        return jsonify({
            "error": "processing_failed",
            "details": str(e)
        }), 500





# Cleanup temp file
        os.remove(tmp_path)
    

        # Feature extraction
        features = extract_features(audio_array=y, sample_rate=sr)

        # Prediction
        prediction = classifier.predict(features)

        # Explanation
        explanation = generate_explanation(features)

        # Response (GUVI compliant)
        return jsonify({
            "classification": prediction["classification"],
            "confidence": round(prediction["confidence"], 3),
            "explanation": explanation
        })

    except Exception as e:
        return jsonify({
            "error": "processing_failed",
            "details": str(e)
        }), 500


# =====================================================
# RUN
# =====================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

