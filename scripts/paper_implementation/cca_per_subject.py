import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/../../../..'))

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.cross_decomposition import CCA
import pickle
from eeg_filterbank import eeg_modulation_filterbank, get_window_lengths_samples

# Parameters
EEG_PATH = 'cocoha-matlab-toolbox-master/python_attention_pipeline/data/eeg_tensor.npy'
AUDIO_ENV_DIR = 'cocoha-matlab-toolbox-master/python_attention_pipeline/data/paper_replication/audio_envelopes/'
OUTPUT_DIR = 'cocoha-matlab-toolbox-master/python_attention_pipeline/data/paper_replication/cca_outputs/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load data
eeg = np.load(EEG_PATH)  # shape: (subjects, trials, channels, time)
labels = np.load('cocoha-matlab-toolbox-master/python_attention_pipeline/data/labels.npy')  # shape: (subjects, trials)

# After loading eeg
if True:
    print(f'eeg.shape = {eeg.shape}')

# Process each subject
n_subjects = eeg.shape[0]
for subj in range(n_subjects):
    print(f"Processing subject {subj+1}/{n_subjects}")
    
    # Get subject's data
    subj_eeg = eeg[subj]  # shape: (trials, channels, time)
    subj_labels = labels[subj]  # shape: (trials,)
    
    # Inside the subject loop, after getting subj_eeg
    if subj == 0:
        print(f'subj_eeg.shape = {subj_eeg.shape}')
    
    # Split trials into train/test (80/20) using stratified split
    train_idx, test_idx = train_test_split(
        np.arange(len(subj_labels)),
        test_size=0.2,
        stratify=subj_labels,
        random_state=42
    )
    
    # Save train/test indices
    np.save(os.path.join(OUTPUT_DIR, f'subj{subj+1}_train_idx.npy'), train_idx)
    np.save(os.path.join(OUTPUT_DIR, f'subj{subj+1}_test_idx.npy'), test_idx)
    
    # Process each trial
    for trial in range(len(subj_labels)):
        # Load audio envelopes for this trial
        env0 = np.load(os.path.join(AUDIO_ENV_DIR, f'S{subj+1}_trial{trial+1}_mixture_0.npy'))
        env1 = np.load(os.path.join(AUDIO_ENV_DIR, f'S{subj+1}_trial{trial+1}_mixture_1.npy'))
        # Get EEG data for this trial
        trial_eeg = subj_eeg[trial]  # shape: (time, channels)
        # Apply filterbank to EEG
        fs = 32
        n_bands = 10
        window_lengths_samples = get_window_lengths_samples(fs, n_bands)
        filtered_eeg = eeg_modulation_filterbank(trial_eeg, fs, window_lengths_samples)  # (time, channels * n_bands)
        # Ensure env1 is (time, 1)
        if env1.ndim == 2 and env1.shape[0] == 1:
            env1 = env1.flatten()
        if env1.ndim == 1:
            env1 = env1.reshape(-1, 1)
        min_len = min(filtered_eeg.shape[0], env1.shape[0])
        if subj == 0 and trial < 5:
            print(f'Trial {trial}: filtered_eeg.shape = {filtered_eeg.shape}, env1.shape = {env1.shape}, min_len = {min_len}')
        filtered_eeg = filtered_eeg[:min_len]
        env1 = env1[:min_len]
        if subj == 0 and trial < 5:
            print(f'Trial {trial}: original env1.shape = {env1.shape}')
        # Perform CCA on filterbanked EEG
        cca = CCA(n_components=1)
        X_c, Y_c = cca.fit_transform(filtered_eeg, env1)
        # Save CCA outputs and both EEG versions
        outputs = {
            'X_c': X_c,
            'Y_c': Y_c,
            'x_weights': cca.x_weights_,
            'y_weights': cca.y_weights_,
            'x_mean': cca._x_mean,
            'y_mean': cca._y_mean,
            'filtered_eeg': filtered_eeg,
            'original_eeg': trial_eeg
        }
        with open(os.path.join(OUTPUT_DIR, f'subj{subj+1}_trial{trial+1}_cca.pkl'), 'wb') as f:
            pickle.dump(outputs, f)
    
    print(f"Saved CCA outputs for subject {subj+1}") 