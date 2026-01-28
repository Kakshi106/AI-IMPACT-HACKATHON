import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# =====================================================
# VoiceClassifier class
# =====================================================

class VoiceClassifier:
    """
    Wrapper class around a Random Forest model
    used to classify AI-generated vs Human voice.

    Responsibilities:
    -----------------
    - Train a model (offline)
    - Load a trained model (runtime)
    - Predict label + confidence (API usage)
    """

    def __init__(self, model_path=None):
        """
        Initialize the classifier.

        If model_path is provided:
        - load an already trained model

        Otherwise:
        - create a fresh untrained model
        """

        if model_path:
            # Load trained model from disk
            self.model = joblib.load(model_path)
        else:
            # Create a new Random Forest model
            self.model = RandomForestClassifier(
                n_estimators=200,        # number of trees
                max_depth=10,            # prevents overfitting
                random_state=42,
                class_weight="balanced"  # handles class imbalance
            )

        self.feature_order = None


# =====================================================
# Training logic (used once, offline)
# =====================================================

    def train(self, feature_dicts, labels):
        """
        Train the classifier.

        Parameters:
        -----------
        feature_dicts : list of dict
            Each dict is output from extract_features()
        labels : list
            0 = Human voice
            1 = AI-generated voice
        """

        # Fix feature order once (VERY IMPORTANT)
        self.feature_order = sorted(feature_dicts[0].keys())

        # Convert dict features → numeric matrix
        X = np.array([
            [f[k] for k in self.feature_order]
            for f in feature_dicts
        ])

        y = np.array(labels)

        # Train the Random Forest
        self.model.fit(X, y)


# =====================================================
# Save / Load model
# =====================================================

    def save(self, model_path):
        """
        Save trained model to disk.
        """
        joblib.dump({
            "model": self.model,
            "feature_order": self.feature_order
        }
        )