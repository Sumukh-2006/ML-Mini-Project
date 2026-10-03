"""Single source of truth for the approved V1 acoustic configuration."""
SAMPLE_RATE = 22050
N_FFT = 1024
WIN_LENGTH = 1024
HOP_LENGTH = 256
N_MELS = 80
F_MIN = 0.0
F_MAX = 8000.0
N_STATES = 3  # 3-state left-to-right monophone HMM
