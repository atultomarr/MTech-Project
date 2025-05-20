import numpy as np

def get_window_lengths_samples(fs, n_bands=10, min_sec=None, max_sec=1.0):
    if min_sec is None:
        min_sec = 1 / fs
    window_lengths_sec = np.logspace(np.log10(min_sec), np.log10(max_sec), n_bands)
    window_lengths_samples = np.round(window_lengths_sec * fs).astype(int)
    window_lengths_samples = np.unique(window_lengths_samples)  # Remove duplicates
    return window_lengths_samples

def eeg_modulation_filterbank(eeg, fs, window_lengths_samples):
    # eeg: (time, channels)
    filtered = []
    for win_len in window_lengths_samples:
        if win_len < 1:
            continue
        kernel = np.ones(win_len) / win_len
        # Convolve each channel
        lowpassed = np.apply_along_axis(lambda x: np.convolve(x, kernel, mode='same'), 0, eeg)
        # Differentiate (first difference)
        diff = np.diff(lowpassed, axis=0, prepend=0)
        filtered.append(diff)
    # Stack along last axis: (time, channels * bands)
    return np.concatenate(filtered, axis=1)

if __name__ == '__main__':
    # Example usage on dummy data
    fs = 32
    n_bands = 10
    eeg = np.random.randn(1000, 64)  # 1000 timepoints, 64 channels
    window_lengths_samples = get_window_lengths_samples(fs, n_bands)
    filtered = eeg_modulation_filterbank(eeg, fs, window_lengths_samples)
    print('Filtered EEG shape:', filtered.shape)  # Should be (1000, 64 * n_bands) 