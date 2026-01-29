import librosa
import numpy as np


def compute_stft(y, n_fft=1024, hop_length=256):
    """
    Compute magnitude STFT.
    """
    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    return np.abs(stft)


def compute_mel_spectrogram(y, sr, n_mels=64, n_fft=1024, hop_length=256):
    """
    Compute Mel spectrogram (power).
    """
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length
    )

    # Convert to log scale (human perception)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db
