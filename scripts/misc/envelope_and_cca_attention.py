import numpy as np
import soundfile as sf
from scipy.signal import hilbert
from sklearn.cross_decomposition import CCA
import matplotlib.pyplot as plt
import os
import sys

# Parameters
subject_idx = 0
trial_idx = 0
fs = 32
n_eeg_samples = 3968

# Paths
EEG_PATH = '/data/home1/rishavs/atul/cocoha-matlab-toolbox-master/cocoha-matlab-toolbox-master/python_attention_pipeline/data/eeg_tensor_filterbank.npy'
AUDIO_DIR = '/data/home1/rishavs/atul/cocoha-matlab-toolbox-master/cocoha-matlab-toolbox-master/python_attention_pipeline/data/aligned_audio/'

subj_str = f'S{subject_idx+1}'
trial_str = f'trial{trial_idx+1}'
audio0_path = os.path.join(AUDIO_DIR, f'{subj_str}_{trial_str}_mixture_0_aligned.wav')
audio1_path = os.path.join(AUDIO_DIR, f'{subj_str}_{trial_str}_mixture_1_aligned.wav')

# Load EEG
eeg = np.load(EEG_PATH, mmap_mode='r')
eeg_trial = eeg[subject_idx, trial_idx]  # (3968, 64)
print(f'Loaded EEG shape: {eeg_trial.shape}')
print('First 5 values of EEG channel 0:', eeg_trial[:5, 0])
nan_count = np.isnan(eeg_trial).sum()
print(f'Number of NaNs in EEG: {nan_count}')
if nan_count > 0:
    print('ERROR: EEG contains NaNs. Exiting.')
    sys.exit(1)

# Remove constant channels before z-scoring
stds = np.std(eeg_trial, axis=0)
print('EEG channel stds:', stds)
nonconst_mask = stds > 0
n_removed = np.sum(~nonconst_mask)
print(f'Removing {n_removed} constant channels (std==0)')
eeg_trial_nonconst = eeg_trial[:, nonconst_mask]

# Load aligned audio
audio0, fs0 = sf.read(audio0_path)
audio1, fs1 = sf.read(audio1_path)
print(f'Loaded audio0 shape: {audio0.shape}, fs: {fs0}, first 5 values: {audio0[:5]}')
print(f'Loaded audio1 shape: {audio1.shape}, fs: {fs1}, first 5 values: {audio1[:5]}')
if fs0 != fs or fs1 != fs:
    print(f'ERROR: Audio sampling rate does not match EEG ({fs}). Exiting.')
    sys.exit(1)
if len(audio0) != n_eeg_samples or len(audio1) != n_eeg_samples:
    print(f'ERROR: Audio and EEG lengths do not match. Exiting.')
    sys.exit(1)

# Envelope extraction
def extract_envelope(audio):
    analytic = hilbert(audio)
    envelope = np.abs(analytic)
    return envelope

env0 = extract_envelope(audio0)
env1 = extract_envelope(audio1)

# Optionally z-score EEG and envelopes for CCA
from scipy.stats import zscore
eeg_z = zscore(eeg_trial_nonconst, axis=0)
env0_z = zscore(env0)
env1_z = zscore(env1)

# Remove all time points (rows) where there is a NaN in any EEG channel
valid_mask = ~np.isnan(eeg_z).any(axis=1)
eeg_z_valid = eeg_z[valid_mask]
env0_z_valid = env0_z[valid_mask]
env1_z_valid = env1_z[valid_mask]

print(f'Number of valid samples after removing NaNs: {eeg_z_valid.shape[0]} (out of {n_eeg_samples})')
if eeg_z_valid.shape[0] < 2:
    print('ERROR: Not enough valid samples for CCA. Exiting.')
    sys.exit(1)

# Run CCA (using all non-constant EEG channels)
cca = CCA(n_components=1)
# CCA expects 2D arrays: (samples, features)
cca.fit(eeg_z_valid, env0_z_valid.reshape(-1, 1))
X_c0, Y_c0 = cca.transform(eeg_z_valid, env0_z_valid.reshape(-1, 1))
corr0 = np.corrcoef(X_c0[:, 0], Y_c0[:, 0])[0, 1]

cca.fit(eeg_z_valid, env1_z_valid.reshape(-1, 1))
X_c1, Y_c1 = cca.transform(eeg_z_valid, env1_z_valid.reshape(-1, 1))
corr1 = np.corrcoef(X_c1[:, 0], Y_c1[:, 0])[0, 1]

print(f'CCA correlation with stream 0: {corr0:.4f}')
print(f'CCA correlation with stream 1: {corr1:.4f}')
attended = np.argmax([corr0, corr1])
print(f'Predicted attended stream: {attended}')

# Optional: plot envelopes and EEG (first channel, valid only)
plt.figure(figsize=(10, 5))
t = np.arange(eeg_z_valid.shape[0]) / fs
plt.plot(t, eeg_z_valid[:, 0], label='EEG (ch 0, z-scored, valid)', alpha=0.7)
plt.plot(t, env0_z_valid, label='Envelope 0 (z-scored, valid)', alpha=0.7)
plt.plot(t, env1_z_valid, label='Envelope 1 (z-scored, valid)', alpha=0.7)
plt.title('EEG and Audio Envelopes (z-scored, valid samples)')
plt.xlabel('Time (s)')
plt.legend()
plt.tight_layout()
plt.show() 