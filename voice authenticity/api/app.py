import os
import subprocess
from uuid import uuid4
from datetime import datetime

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    send_from_directory,
    jsonify,
)

from werkzeug.utils import secure_filename

# NEW imports (replace fingerprint logic)
from voice_features.extract_features import extract_features
from models.voice_classifier import VoiceClassifier
from explain.explanation import generate_explanation


# ----- Flask setup -----
app = Flask(__name__)
app.secret_key = "super-secret-key"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {"wav", "webm", "ogg"}
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load trained model ONCE
classifier = VoiceClassifier.load("voice_model.pkl")


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


# -------------------------------------------------
# MAIN PAGE (UPLOAD FILE)
# -------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "clip" not in request.files:
            flash("No file uploaded.")
            return redirect(request.url)

        file = request.files["clip"]
        if file.filename == "":
            flash("No selected file.")
            return redirect(request.url)

        if not allowed_file(file.filename):
            flash("Unsupported file type.")
            return redirect(request.url)

        original_name = secure_filename(file.filename)
        ext = original_name.rsplit(".", 1)[1].lower()
        unique_base = uuid4().hex

        raw_path = os.path.join(
            app.config["UPLOAD_FOLDER"],
            f"{unique_base}_raw.{ext}"
        )
        file.save(raw_path)

        wav_path = os.path.join(
            app.config["UPLOAD_FOLDER"],
            f"{unique_base}.wav"
        )

        # Convert to mono 16k wav
        if ext != "wav":
            cmd = [
                "ffmpeg", "-y",
                "-i", raw_path,
                "-ac", "1",
                "-ar", "16000",
                wav_path,
            ]
            subprocess.run(cmd, check=True,
                           stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)
        else:
            os.rename(raw_path, wav_path)

        # -----------------------------
        # NEW AI VOICE DETECTION LOGIC
        # -----------------------------
        features = extract_features(audio_path=wav_path)
        prediction = classifier.predict(features)
        explanation = generate_explanation(features)

        result = {
            "classification": prediction["classification"],
            "confidence": prediction["confidence"],
            "explanation": explanation
        }

        return render_template(
            "result.html",
            filename=original_name,
            audio_url=url_for("uploaded_file", filename=os.path.basename(wav_path)),
            result=result
        )

    return render_template("index.html")


# -------------------------------------------------
# MICROPHONE RECORDING ENDPOINT
# -------------------------------------------------
@app.route("/record", methods=["POST"])
def record():
    if "clip" not in request.files:
        return "No audio received", 400

    file = request.files["clip"]
    unique_base = uuid4().hex

    raw_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        f"{unique_base}_raw.webm"
    )
    file.save(raw_path)

    wav_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        f"{unique_base}.wav"
    )

    cmd = [
        "ffmpeg", "-y",
        "-i", raw_path,
        "-ac", "1",
        "-ar", "16000",
        wav_path,
    ]
    subprocess.run(cmd, check=True,
                   stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL)

    features = extract_features(audio_path=wav_path)
    prediction = classifier.predict(features)
    explanation = generate_explanation(features)

    return jsonify({
        "classification": prediction["classification"],
        "confidence": round(prediction["confidence"], 3),
        "explanation": explanation
    })


if __name__ == "__main__":
    app.run(debug=True)
