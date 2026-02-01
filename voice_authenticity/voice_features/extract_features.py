import numpy as np
import librosa

# =====================================================
# Main entry point
# =====================================================

def extract_features(
    audio_path=None,
    audio_array=None,
    sample_rate=16000
):
    """
    Main function used by the system.

    Purpose:
    --------
    Takes an audio input and extracts language-agnostic
    acoustic features that help distinguish
    AI-generated voice vs human voice.

    Returns:
    --------
    A flat dictionary of numeric features suitable
    for ML classification.
    """

    # -------------------------------------------------
    # 1. Load and normalize audio
    # -------------------------------------------------
    y, sr = load_audio(audio_path, audio_array, sample_rate)

    # SAFETY CHECK:
    # If audio is too short (< 0.5 sec), return safe defaults
    if len(y) < sr * 0.5:
        return default_features()

    features = {}

    # -------------------------------------------------
    # 2. Pitch stability features
    # -------------------------------------------------
    features.update(pitch_features(y, sr))

    # -------------------------------------------------
    # 3. Spectral smoothness & dynamics
    # -------------------------------------------------
    features.update(spectral_features(y, sr))

    # -------------------------------------------------
    # 4. Temporal / energy consistency
    # -------------------------------------------------
    features.update(temporal_features(y))

    # -------------------------------------------------
    # 5. Harmonic vs noise realism
    # -------------------------------------------------
    features.update(harmonic_noise_features(y))

    # -------------------------------------------------
    # 6. Final safety cleanup (avoid NaN / inf)
    # -------------------------------------------------
    for k, v in features.items():
        if np.isnan(v) or np.isinf(v):
            features[k] = 0.0

    return features


# =====================================================
# Default feature vector (used for invalid / short audio)
# =====================================================

def default_features():
    """
    Returns a safe default feature vector.
    Prevents crashes when audio is silent or too short.
    """
    return {
        "pitch_mean": 0.0,
        "pitch_std": 0.0,
        "pitch_delta_std": 0.0,
        "centroid_mean": 0.0,
        "centroid_std": 0.0,
        "rolloff_std": 0.0,
        "bandwidth_std": 0.0,
        "spectral_flux_std": 0.0,
        "energy_mean": 0.0,
        "energy_std": 0.0,
        "energy_delta_std": 0.0,
        "hnr": 0.0
    }


# =====================================================
# Audio loading & normalization
# =====================================================

def load_audio(audio_path, audio_array, target_sr):
    """
    Loads audio either from:
    - a file path, OR
    - a numpy array (already decoded)

    Ensures:
    - mono audio
    - fixed sample rate
    - normalized amplitude
    """

    if audio_array is not None:
        y = audio_array
        sr = target_sr
    else:
        y, sr = librosa.load(audio_path, sr=target_sr, mono=True)

    # Normalize amplitude to [-1, 1]
    max_val = np.max(np.abs(y))
    if max_val > 0:
        y = y / max_val

    return y, sr


# =====================================================
# Pitch stability features
# =====================================================

def pitch_features(y, sr):
    """
    Measures pitch behavior over time.

    Key insight:
    - Human speech has micro pitch variations
    - AI speech is unnaturally stable
    """

    # Extract fundamental frequency (pitch) per frame
    f0 = librosa.yin(
        y,
        fmin=50,    # lower bound (safe for Indian voices)
        fmax=350,   # upper bound (female / expressive speech)
        sr=sr
    )

    # Remove invalid pitch values
    f0 = f0[~np.isnan(f0)]

    # If pitch tracking failed, return safe values
    if len(f0) < 10:
        return {
            "pitch_mean": 0.0,
            "pitch_std": 0.0,
            "pitch_delta_std": 0.0
        }

    # Frame-to-frame pitch change
    pitch_delta = np.diff(f0)

    return {
        # Average pitch (not very important, but useful)
        "pitch_mean": float(np.mean(f0)),

        # Pitch variation (VERY IMPORTANT feature)
        "pitch_std": float(np.std(f0)),

        # Micro pitch movement over time
        "pitch_delta_std": float(np.std(pitch_delta))
    }


# =====================================================
# Spectral features (smoothness & dynamics)
# =====================================================

def spectral_features(y, sr):
    """
    Measures how the frequency content of speech
    changes over time.

    AI voices tend to be over-smooth.
    """

    # Compute Short-Time Fourier Transform
    S = np.abs(librosa.stft(y, n_fft=1024, hop_length=256))

    # Standard spectral descriptors
    centroid = librosa.feature.spectral_centroid(S=S, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(S=S, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(S=S, sr=sr)[0]

    # Spectral flux = frame-to-frame spectral change
    flux = np.sqrt(np.mean(np.diff(S, axis=1) ** 2, axis=0))

    return {
        "centroid_mean": float(np.mean(centroid)),
        "centroid_std": float(np.std(centroid)),
        "rolloff_std": float(np.std(rolloff)),
        "bandwidth_std": float(np.std(bandwidth)),

        # LOW flux → suspiciously smooth (AI)
        "spectral_flux_std": float(np.std(flux))
    }


# =====================================================
# Temporal / energy features
# =====================================================

def temporal_features(y):
    """
    Measures loudness variation over time.

    Humans:
    - pauses
    - breathing
    - emphasis

    AI:
    - uniform energy
    """

    frame_length = 1024
    hop_length = 256

    # Short-time energy (RMS)
    energy = librosa.feature.rms(
        y=y,
        frame_length=frame_length,
        hop_length=hop_length
    )[0]

    energy_delta = np.diff(energy)

    return {
        "energy_mean": float(np.mean(energy)),
        "energy_std": float(np.std(energy)),
        "energy_delta_std": float(np.std(energy_delta))
    }


# =====================================================
# Harmonic vs noise features
# =====================================================

def harmonic_noise_features(y):
    """
    Separates harmonic (voiced) and noise components.

    AI speech often has:
    - unnaturally high harmonic dominance
    """

    harmonic, noise = librosa.effects.hpss(y)

    harmonic_energy = np.sum(harmonic ** 2)
    noise_energy = np.sum(noise ** 2) + 1e-6  # avoid division by zero

    # Harmonic-to-Noise Ratio (HNR)
    hnr = 10 * np.log10(harmonic_energy / noise_energy)

    return {
        "hnr": float(hnr)
    }
