"""Phase: Feature Extraction & Oracle Reconstruction (Gate 3).

Strict V1 Configuration:
- sample_rate: 22050 Hz
- n_fft: 1024
- win_length: 1024
- hop_length: 256
- n_mels: 80
- f_min: 0.0 Hz
- f_max: 8000.0 Hz
- Log magnitude: log(clamp(Mel, min=1e-5))
- Inversion: exp(log_mel) -> pseudo-inverse mel basis -> fast Griffin-Lim (60 iters, momentum=0.99)
- No PCA, no deltas.
"""
import os
import sys
import wave
import numpy as np
import soundfile as sf
import librosa

from preprocessing.config import SAMPLE_RATE, N_FFT, WIN_LENGTH, HOP_LENGTH, N_MELS, F_MIN, F_MAX

# Mel filterbank basis and its pseudo-inverse
MEL_BASIS = librosa.filters.mel(
    sr=SAMPLE_RATE,
    n_fft=N_FFT,
    n_mels=N_MELS,
    fmin=F_MIN,
    fmax=F_MAX,
    norm="slaney",
    htk=False,
)
INV_MEL_BASIS = np.linalg.pinv(MEL_BASIS)


def wav_to_log_mel(wav: np.ndarray) -> np.ndarray:
    """Compute 80-D log-Mel spectrogram.
    wav: 1D float array normalized to [-1.0, 1.0]
    Returns: [T, 80] array of log magnitude
    """
    # STFT with center=True (default standard)
    stft = librosa.stft(
        wav,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        window="hann",
        center=True,
        pad_mode="reflect",
    )
    # Linear magnitude
    mag = np.abs(stft)  # [1 + n_fft//2, T] = [513, T]
    # Filterbank projection: [80, 513] @ [513, T] = [80, T]
    mel = np.dot(MEL_BASIS, mag)
    # Numerical stability clamp before log
    mel = np.maximum(mel, 1e-5)
    log_mel = np.log(mel)
    # Transpose to [T, 80]
    return log_mel.T


def log_mel_to_wav(log_mel: np.ndarray, n_iter: int = 60, momentum: float = 0.99) -> np.ndarray:
    """Invert [T, 80] log-Mel spectrogram back to waveform via pseudo-inverse & fast Griffin-Lim.
    log_mel: [T, 80] float array
    Returns: 1D float array audio
    """
    # [80, T]
    mel = np.exp(log_mel.T)
    # Approximate linear magnitude via pseudo-inverse: [513, 80] @ [80, T] -> [513, T]
    approx_mag = np.dot(INV_MEL_BASIS, mel)
    approx_mag = np.maximum(approx_mag, 1e-8)

    # Fast Griffin-Lim phase estimation
    recon_wav = librosa.griffinlim(
        approx_mag,
        n_iter=n_iter,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        window="hann",
        center=True,
        momentum=momentum,
        random_state=42,
    )
    return recon_wav
