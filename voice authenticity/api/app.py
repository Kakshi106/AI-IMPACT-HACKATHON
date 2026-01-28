import base64
import io
import tempfile
from flask import Flask, request, jsonify

from voice_features.extract_features import extract_features
from models.voice_classifier import VoiceClassifier
from explain.explanation import generate_explanation

import librosa

# =====================================================
# Flask App Setup
# =====================================================

app = Flask(__name__)

# -----------------------------------------------------
# Load trained classifier ONCE at startup
# -----------------------------------------------------

MODEL_PATH = "voice_model.pkl"
classifier = VoiceClassifier.load(MODEL_PATH)

# =====================================================
# Utility: Decode Base64 MP3 → Audio array
# =====================================================

def decode_base64_audio(base64_string, target_sr=16000):
    """
    Decodes Base64-encoded MP3 into audio array.
    """
    audio_bytes = base64.b64decode(base64_string)

    with tempfile.NamedTemporaryFile(suffix=".mp3") as tmp:
        tmp.write(audio_bytes)
        tmp.flush()

        y, sr = librosa.load(tmp.name, sr=target_sr, mono=True)

    return y, sr


# =====================================================
# API Endpoint
# =====================================================

@app.route("/detect", methods=["POST"])
def detect_voice():
    """
    API endpoint for AI-generated voice detection.
    """

    try:
        data = request.get_json()

        if "audio_base64" not in data:
            return jsonify({"error": "audio_base64 field missing"}), 400

        # -------------------------------------------------
        # 1. Decode audio
        # -------------------------------------------------
        audio_array, sr = decode_base64_audio(
            data["audio_base64"]
        )

        # -------------------------------------------------
        # 2. Feature extraction
        # -------------------------------------------------
        features = extract_features(
            audio_array=audio_array,
            sample_rate=sr
        )

        # -------------------------------------------------
        # 3. Classification
        # -------------------------------------------------
        prediction = classifier.predict(features)

        # -------------------------------------------------
        # 4. Explanation
        # -------------------------------------------------
        explanation = generate_explanation(features)

        # -------------------------------------------------
        # 5. Final response
        # -------------------------------------------------
        response = {
            "classification": prediction["classification"],
            "confidence": round(prediction["confidence"], 3),
            "explanation": explanation
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({
            "error": "Internal server error",
            "details": str(e)
        }), 500


# =====================================================
# Run Server
# =====================================================

if __name__ == "__main__":
    app.run(debug=True)
