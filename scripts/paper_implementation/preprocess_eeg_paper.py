import numpy as np
from scipy import signal
from scipy.io import loadmat
import os
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
from sklearn.model_selection import train_test_split

# Parameters from the paper
PAPER_PARAMS = {
    'lowpass_freq': 20,  # Hz
    'highpass_freq': 0.1,  # Hz
    'target_fs': 64,  # Hz (matching audio sampling rate)
    'lowpass_order': 4,  # 4th order Butterworth
    'highpass_order': 2,  # 2nd order Butterworth
    'pca_variance': 0.99,  # Keep components explaining 99% of variance
    'n_cca_components': 1  # Number of CCA components to use
}

# Dyadic filterbank parameters (copied from audio_envelope_paper.py)
WINDOW_LENGTHS = np.logspace(np.log10(1/32), np.log10(1), 10)  # in seconds
DOWNSAMPLED_FS = 64  # Hz

# Paths
MAT_DIR = 'python_attention_pipeline/data/mat_subjects'
PROCESSED_AUDIO_DIR = 'python_attention_pipeline/data/processed_audio'

def create_butterworth_filter(freq, order, fs, btype):
    """Create Butterworth filter coefficients."""
    nyquist = fs / 2
    normalized_freq = freq / nyquist
    b, a = signal.butter(order, normalized_freq, btype=btype)
    return b, a

def apply_common_average_reference(eeg_data):
    """Apply common average reference to EEG data."""
    return eeg_data - np.mean(eeg_data, axis=1, keepdims=True)

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

def preprocess_eeg_trial(eeg_data, original_fs, pca_model=None, cca_model=None, channel_mask=None):
    """Preprocess a single trial of EEG data according to paper specifications."""
    # Ensure exactly 64 channels
    if eeg_data.shape[1] < 64:
        # Pad with zeros if fewer than 64 channels
        padded = np.zeros((eeg_data.shape[0], 64))
        padded[:, :eeg_data.shape[1]] = eeg_data
        eeg_data = padded
    elif eeg_data.shape[1] > 64:
        # Use only the first 64 channels if more than 64
        eeg_data = eeg_data[:, :64]
    
    # Apply channel mask if provided
    if channel_mask is not None:
        eeg_data = eeg_data[channel_mask]
    
    # 1. Low-pass filter at 20 Hz (4th order Butterworth)
    b_low, a_low = create_butterworth_filter(
        PAPER_PARAMS['lowpass_freq'],
        PAPER_PARAMS['lowpass_order'],
        original_fs,
        'low'
    )
    eeg_data = signal.filtfilt(b_low, a_low, eeg_data, axis=0)
    
    # 2. Downsample to 64 Hz
    n_samples = int(len(eeg_data) * PAPER_PARAMS['target_fs'] / original_fs)
    eeg_data = signal.resample(eeg_data, n_samples, axis=0)
    
    # 3. High-pass filter at 0.1 Hz (2nd order Butterworth)
    b_high, a_high = create_butterworth_filter(
        PAPER_PARAMS['highpass_freq'],
        PAPER_PARAMS['highpass_order'],
        PAPER_PARAMS['target_fs'],
        'high'
    )
    eeg_data = signal.filtfilt(b_high, a_high, eeg_data, axis=0)
    
    # 4. Apply common average reference
    eeg_data = apply_common_average_reference(eeg_data)

    # 5. Apply dyadic filterbank to each channel
    n_channels = eeg_data.shape[1]
    n_bands = len(WINDOW_LENGTHS)
    filtered_eeg = []
    for ch in range(n_channels):
        # Apply dyadic filterbank to each channel
        bands = dyadic_filterbank(eeg_data[:, ch], DOWNSAMPLED_FS)  # shape: (n_bands, n_samples)
        filtered_eeg.append(bands)
    # Stack: shape (n_channels, n_bands, n_samples) -> (n_samples, n_channels * n_bands)
    filtered_eeg = np.stack(filtered_eeg, axis=1)  # (n_bands, n_channels, n_samples)
    filtered_eeg = np.transpose(filtered_eeg, (2, 1, 0))  # (n_samples, n_channels, n_bands)
    filtered_eeg = filtered_eeg.reshape(filtered_eeg.shape[0], -1)  # (n_samples, n_channels * n_bands)
    eeg_data = filtered_eeg

    # 6. Apply PCA transformation
    if pca_model is None:
        # If no PCA model provided, create one
        pca_model = PCA(n_components=PAPER_PARAMS['pca_variance'])
        eeg_data = pca_model.fit_transform(eeg_data)
    else:
        # Use existing PCA model
        eeg_data = pca_model.transform(eeg_data)
    
    # 7. Apply CCA transformation if model is provided
    if cca_model is not None:
        eeg_data = cca_model.transform(eeg_data)
    
    return eeg_data, pca_model

def main():
    # Input paths
    mat_dir = 'python_attention_pipeline/data/mat_subjects'
    output_dir = 'python_attention_pipeline/data'
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of all .mat files
    mat_files = sorted([f for f in os.listdir(mat_dir) if f.startswith('S') and f.endswith('.mat')])
    
    # First pass: collect all data for PCA and CCA training
    print("First pass: Collecting data for PCA and CCA training...")
    all_eeg_data = []
    all_audio_data = []
    channel_counts = []
    
    # First, collect channel counts from all trials
    for mat_file in tqdm(mat_files, desc="Collecting channel counts"):
        mat_path = os.path.join(mat_dir, mat_file)
        data = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
        
        if 'preproc_trials' in data:
            trials = data['preproc_trials']
        elif 'trials' in data:
            trials = data['trials']
        else:
            continue
        
        for trial in trials:
            eeg_data = trial.RawData.EegData
            channel_counts.append(eeg_data.shape[0])
    
    # Find the minimum number of channels
    min_channels = min(channel_counts)
    print(f"Found minimum number of channels: {min_channels}")
    
    # Now process the data using only the first min_channels channels
    for mat_file in tqdm(mat_files, desc="Collecting data"):
        mat_path = os.path.join(mat_dir, mat_file)
        data = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
        
        if 'preproc_trials' in data:
            trials = data['preproc_trials']
        elif 'trials' in data:
            trials = data['trials']
        else:
            continue
        
        for trial_idx, trial in enumerate(trials):
            eeg_data = trial.RawData.EegData
            original_fs = trial.FileHeader.SampleRate
            
            # Basic preprocessing without PCA/CCA, using only first min_channels channels
            processed_eeg, _ = preprocess_eeg_trial(
                eeg_data[:min_channels],
                original_fs
            )
            all_eeg_data.append(processed_eeg)
            
            # Load processed audio envelopes
            subject_num = mat_file[1:-4]  # Extract subject number
            trial_num = trial_idx + 1
            
            env0_path = os.path.join(PROCESSED_AUDIO_DIR, f'S{subject_num}_trial{trial_num}_envelopes_0.npy')
            env1_path = os.path.join(PROCESSED_AUDIO_DIR, f'S{subject_num}_trial{trial_num}_envelopes_1.npy')
            
            if os.path.exists(env0_path) and os.path.exists(env1_path):
                env0 = np.load(env0_path)
                env1 = np.load(env1_path)
                all_audio_data.append((env0, env1))
    
    # Fit PCA on all collected data
    print("Fitting PCA on all data...")
    all_eeg_data = np.vstack(all_eeg_data)
    pca_model = PCA(n_components=PAPER_PARAMS['pca_variance'])
    pca_model.fit(all_eeg_data)
    print(f"PCA: Reduced from {all_eeg_data.shape[1]} to {pca_model.n_components_} components")
    
    # If audio data is available, fit CCA
    cca_model = None
    if all_audio_data:
        print("Fitting CCA on training data...")
        # Split data into training and validation sets
        train_idx, val_idx = train_test_split(
            range(len(all_eeg_data)), 
            test_size=0.2, 
            random_state=42
        )
        
        # Prepare audio data for CCA
        audio_envs = np.vstack([env0 for env0, _ in all_audio_data])
        
        # Fit CCA on training data
        cca_model = CCA(n_components=PAPER_PARAMS['n_cca_components'])
        cca_model.fit(
            pca_model.transform(all_eeg_data[train_idx]),
            audio_envs[train_idx].reshape(-1, 1)
        )
    
    # Second pass: process all data using the trained models
    print("Second pass: Processing all data with trained models...")
    processed_trials = []
    subject_trial_count = []
    
    for mat_file in tqdm(mat_files, desc="Processing subjects"):
        mat_path = os.path.join(mat_dir, mat_file)
        data = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
        
        if 'preproc_trials' in data:
            trials = data['preproc_trials']
        elif 'trials' in data:
            trials = data['trials']
        else:
            continue
        
        subject_trials = []
        for trial in trials:
            eeg_data = trial.RawData.EegData
            original_fs = trial.FileHeader.SampleRate
            
            # Process with trained models, using only first min_channels channels
            processed_eeg, _ = preprocess_eeg_trial(
                eeg_data[:min_channels], 
                original_fs,
                pca_model=pca_model,
                cca_model=cca_model
            )
            subject_trials.append(processed_eeg)
        
        processed_trials.append(subject_trials)
        subject_trial_count.append(len(subject_trials))
    
    # Convert to numpy array
    n_subjects = len(processed_trials)
    n_trials = max(subject_trial_count)
    n_channels = pca_model.n_components_
    if cca_model:
        n_channels = PAPER_PARAMS['n_cca_components']
    
    # Get maximum time points
    max_time = max(trial.shape[0] for subject in processed_trials for trial in subject)
    
    # Create the tensor
    eeg_tensor = np.full((n_subjects, n_trials, max_time, n_channels), np.nan)
    
    # Fill the tensor
    for i, subject_trials in enumerate(processed_trials):
        for j, trial in enumerate(subject_trials):
            eeg_tensor[i, j, :trial.shape[0], :] = trial
    
    # Save the processed data
    output_path = os.path.join(output_dir, 'eeg_tensor_paper.npy')
    np.save(output_path, eeg_tensor)
    print(f"Saved processed EEG tensor to {output_path}")
    print(f"Tensor shape: {eeg_tensor.shape}")
    print(f"Sampling rate: {PAPER_PARAMS['target_fs']} Hz")
    print(f"Number of components: {n_channels}")

if __name__ == "__main__":
    main() 