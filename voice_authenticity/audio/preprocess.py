import librosa
import numpy as np


def load_and_preprocess(audio_path, target_sr=16000):
    """
    Load audio file and apply standard preprocessing.

    Steps:
    - Load audio
    - Convert to mono
    - Resample to target_sr
    - Normalize amplitude

    Returns:
    - y: numpy array (audio waveform)
    - sr: sample rate
    """

    # Load audio (librosa handles many formats)
    y, sr = librosa.load(audio_path, sr=target_sr, mono=True)

    # Normalize audio amplitude
    max_val = np.max(np.abs(y))
    if max_val > 0:
        y = y / max_val

    return y, sr
