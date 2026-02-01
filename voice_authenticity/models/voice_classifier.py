import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier


class VoiceClassifier:
    """
    Random Forest based classifier for
    AI-generated vs Human voice detection.
    """

    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=42,
            class_weight="balanced"
        )
        self.feature_order = None

    # -----------------------------
    # TRAINING (offline)
    # -----------------------------
    def train(self, feature_dicts, labels):
        """
        Train the classifier.

        feature_dicts : list of dict
        labels : list (0 = Human, 1 = AI)
        """

        # Fix feature order
        self.feature_order = sorted(feature_dicts[0].keys())

        X = np.array([
            [f[k] for k in self.feature_order]
            for f in feature_dicts
        ])
        y = np.array(labels)

        self.model.fit(X, y)

    # -----------------------------
    # SAVE MODEL
    # -----------------------------
    def save(self, model_path):
        """
        Save trained model to disk.
        """
        joblib.dump({
            "model": self.model,
            "feature_order": self.feature_order
        }, model_path)

    # -----------------------------
    # LOAD MODEL (STATIC)
    # -----------------------------
    @staticmethod
    def load(model_path):
        """
        Load trained model from disk.
        """
        data = joblib.load(model_path)

        classifier = VoiceClassifier()
        classifier.model = data["model"]
        classifier.feature_order = data["feature_order"]

        return classifier

    # -----------------------------
    # PREDICTION (API usage)
    # -----------------------------
    def predict(self, feature_dict):
        """
        Predict AI vs Human voice.
        """

        X = np.array([
            feature_dict[k] for k in self.feature_order
        ]).reshape(1, -1)

        probs = self.model.predict_proba(X)[0]
        ai_prob = probs[1]

        return {
            "classification": "AI_GENERATED" if ai_prob >= 0.5 else "HUMAN",
            "confidence": float(ai_prob)
        }
