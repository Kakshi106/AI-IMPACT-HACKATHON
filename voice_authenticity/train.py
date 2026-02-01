import os
import glob

from voice_authenticity.voice_features.extract_features import extract_features
from voice_authenticity.models.voice_classifier import VoiceClassifier


# =====================================================
# PATH CONFIG (FIXED & SAFE)
# =====================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
HUMAN_DIR = os.path.join(DATA_DIR, "human")
AI_DIR = os.path.join(DATA_DIR, "ai")

MODEL_PATH = os.path.join(BASE_DIR, "voice_model.pkl")



# =====================================================
# LOAD DATASET
# =====================================================

def load_dataset():

    print("HUMAN DIR:", HUMAN_DIR)
    print("AI DIR:", AI_DIR)
    print("HUMAN FILES:", glob.glob(os.path.join(HUMAN_DIR, "*.wav")))
    print("AI FILES:", glob.glob(os.path.join(AI_DIR, "*.wav")))

    feature_list = []
    labels = []

    print("Looking for HUMAN files in:", HUMAN_DIR)
    human_files = glob.glob(os.path.join(HUMAN_DIR, "*.wav"))
    print("Found HUMAN files:", human_files)

    print("Looking for AI files in:", AI_DIR)
    ai_files = glob.glob(os.path.join(AI_DIR, "*.wav"))
    print("Found AI files:", ai_files)

    for audio_path in human_files:
        features = extract_features(audio_path=audio_path)
        feature_list.append(features)
        labels.append(0)

    for audio_path in ai_files:
        features = extract_features(audio_path=audio_path)
        feature_list.append(features)
        labels.append(1)

    return feature_list, labels



# =====================================================
# TRAIN MODEL
# =====================================================

def train():
    print("📥 Loading dataset...")
    features, labels = load_dataset()

    if len(features) < 4:
        raise RuntimeError(
            "Not enough data. Add at least a few human and AI samples."
        )

    print("🧠 Training classifier...")
    classifier = VoiceClassifier()
    classifier.train(features, labels)

    print("💾 Saving model...")
    classifier.save(MODEL_PATH)

    print(f"✅ Training complete. Model saved as {MODEL_PATH}")


# =====================================================
# ENTRY POINT
# =====================================================

if __name__ == "__main__":
    train()
