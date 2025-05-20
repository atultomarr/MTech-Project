import os
import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, resample
from tqdm import tqdm

# Parameters
EEG_DIR = 'python_attention_pipeline/data/mat_subjects'
OUTPUT_PATH = 'python_attention_pipeline/data/paper_replication/eeg_tensor_paper.npy'

LOWPASS_FREQ = 20  # Hz
LOWPASS_ORDER = 4
HIGHPASS_FREQ = 0.1  # Hz
HIGHPASS_ORDER = 2
DOWNSAMPLED_FS = 64  # Hz

# Dyadic filterbank parameters
WINDOW_LENGTHS = np.logspace(np.log10(1/32), np.log10(1), 10)  # in seconds


def butter_filter(data, cutoff, fs, order, btype):
    b, a = butter(order, cutoff / (fs / 2), btype=btype)
    return filtfilt(b, a, data, axis=0)


def common_average_reference(eeg):
    return eeg - np.mean(eeg, axis=1, keepdims=True)


def dyadic_filterbank(signal, fs):
    """
    Apply a dyadic filterbank: convolve with square windows of various lengths, then differentiate.
    Returns a (n_bands, n_samples, n_channels) array for multichannel input.
    """
    n_samples, n_channels = signal.shape
    filtered = []
    for win_sec in WINDOW_LENGTHS:
        win_len = int(round(win_sec * fs))
        if win_len < 1:
            win_len = 1
        kernel = np.ones(win_len) / win_len
        conv = np.apply_along_axis(lambda x: np.convolve(x, kernel, mode='same'), 0, signal)
        # First-order differentiation
        diff = np.diff(conv, axis=0, prepend=conv[0:1, :])
        filtered.append(diff)
    return np.stack(filtered, axis=0)  # (n_bands, n_samples, n_channels)


def preprocess_eeg_trial(eeg, fs):
    # 1. Low-pass filter
    eeg = butter_filter(eeg, LOWPASS_FREQ, fs, LOWPASS_ORDER, 'low')
    # 2. Downsample
    n_samples = int(eeg.shape[0] * DOWNSAMPLED_FS / fs)
    eeg = resample(eeg, n_samples, axis=0)
    # 3. High-pass filter
    eeg = butter_filter(eeg, HIGHPASS_FREQ, DOWNSAMPLED_FS, HIGHPASS_ORDER, 'high')
    # 4. Common average reference
    eeg = common_average_reference(eeg)
    # 5. Dyadic filterbank
    features = dyadic_filterbank(eeg, DOWNSAMPLED_FS)
    return features  # (n_bands, n_samples, n_channels)


def main():
    mat_files = [f for f in os.listdir(EEG_DIR) if f.endswith('.mat')]
    mat_files.sort()
    all_subjects = []
    max_trials = 0
    max_samples = 0
    n_bands = len(WINDOW_LENGTHS)
    n_channels = None
    # First pass: preprocess and collect shapes
    for fname in tqdm(mat_files, desc='Preprocessing EEG (collect shapes)'):
        mat_path = os.path.join(EEG_DIR, fname)
        data = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
        if 'preproc_trials' in data:
            trials = data['preproc_trials']
        elif 'trials' in data:
            trials = data['trials']
        else:
            continue
        subject_trials = []
        for trial in trials:
            eeg = trial.RawData.EegData
            fs = trial.FileHeader.SampleRate
            features = preprocess_eeg_trial(eeg, fs)
            subject_trials.append(features)
            max_samples = max(max_samples, features.shape[1])
            if n_channels is None:
                n_channels = features.shape[2]
        all_subjects.append(subject_trials)
        max_trials = max(max_trials, len(subject_trials))
    n_subjects = len(all_subjects)
    # Allocate tensor
    eeg_tensor = np.full((n_subjects, max_trials, n_bands, max_samples, n_channels), np.nan, dtype=np.float32)
    # Fill tensor
    for i, subject_trials in enumerate(all_subjects):
        for j, features in enumerate(subject_trials):
            eeg_tensor[i, j, :, :features.shape[1], :] = features
    # Save
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    np.save(OUTPUT_PATH, eeg_tensor)
    print(f'Saved EEG tensor to {OUTPUT_PATH} with shape {eeg_tensor.shape}')

if __name__ == '__main__':
    main() 