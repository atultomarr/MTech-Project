import os
import numpy as np
import soundfile as sf
from scipy.signal import butter, filtfilt, decimate
from tqdm import tqdm

# Parameters
AUDIO_DIR = 'python_attention_pipeline/data/separated'
OUTPUT_DIR = 'python_attention_pipeline/data/paper_replication/audio_envelopes'
os.makedirs(OUTPUT_DIR, exist_ok=True)

LOWPASS_FREQ = 20  # Hz
LOWPASS_ORDER = 4
DOWNSAMPLED_FS = 64  # Hz
COMPRESSIVE_POWER = 0.3

# Dyadic filterbank parameters
WINDOW_LENGTHS = np.logspace(np.log10(1/32), np.log10(1), 10)  # in seconds


def butter_lowpass_filter(data, cutoff, fs, order):
    b, a = butter(order, cutoff / (fs / 2), btype='low')
    return filtfilt(b, a, data)


def dyadic_filterbank(signal, fs):
    """
    Apply a dyadic filterbank: convolve with square windows of various lengths, then differentiate.
    Returns a (n_windows, n_samples) array.
    """
    filtered = []
    for win_sec in WINDOW_LENGTHS:
        win_len = int(round(win_sec * fs))
        if win_len < 1:
            win_len = 1
        kernel = np.ones(win_len) / win_len
        conv = np.convolve(signal, kernel, mode='same')
        # First-order differentiation
        diff = np.diff(conv, prepend=conv[0])
        filtered.append(diff)
    return np.stack(filtered, axis=0)  # shape: (n_bands, n_samples)


def process_audio_file(audio_path, output_path):
    audio, fs = sf.read(audio_path)
    if audio.ndim > 1:
        audio = audio[:, 0]  # Use first channel if stereo
    # 1. Square
    audio = audio ** 2
    audio = np.nan_to_num(audio)
    # 2. Low-pass filter
    audio = butter_lowpass_filter(audio, LOWPASS_FREQ, fs, LOWPASS_ORDER)
    audio = np.nan_to_num(audio)
    # 3. Downsample to 64 Hz
    decim_factor = int(round(fs / DOWNSAMPLED_FS))
    audio = decimate(audio, decim_factor, ftype='fir', zero_phase=True)
    audio = np.nan_to_num(audio)
    # 4. Power law compression (use abs to avoid NaNs)
    audio = np.abs(audio) ** COMPRESSIVE_POWER
    audio = np.nan_to_num(audio)
    # 5. Dyadic filterbank
    features = dyadic_filterbank(audio, DOWNSAMPLED_FS)
    features = np.nan_to_num(features)
    # Save as .npy
    np.save(output_path, features)


def main():
    audio_files = [f for f in os.listdir(AUDIO_DIR) if f.endswith('.wav') or f.endswith('.flac')]
    for fname in tqdm(audio_files, desc='Processing audio files'):
        in_path = os.path.join(AUDIO_DIR, fname)
        out_path = os.path.join(OUTPUT_DIR, fname.replace('.wav', '.npy').replace('.flac', '.npy'))
        process_audio_file(in_path, out_path)

if __name__ == '__main__':
    main() 