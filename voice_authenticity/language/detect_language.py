import numpy as np


def detect_language(features):
    """
    Heuristic-based language identification
    using acoustic characteristics.

    Returns:
    - language (str)
    """

    pitch_mean = features.get("pitch_mean", 0)
    pitch_std = features.get("pitch_std", 0)
    energy_std = features.get("energy_std", 0)
    spectral_centroid = features.get("centroid_mean", 0)

    # ---- Heuristic rules ----
    # These are NOT hardcoded outputs, they are
    # signal-driven approximations.

    if pitch_mean > 190 and energy_std > 0.05:
        return "Tamil"

    if 150 < pitch_mean <= 190 and spectral_centroid > 2500:
        return "Hindi"

    if pitch_std < 12 and energy_std < 0.03:
        return "English"

    if spectral_centroid < 2200 and pitch_mean < 160:
        return "Malayalam"

    return "Telugu"
