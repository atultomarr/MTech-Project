import numpy as np
import soundfile as sf
import os
import matplotlib.pyplot as plt

# Paths
EEG_PATH = '/data/home1/rishavs/atul/cocoha-matlab-toolbox-master/cocoha-matlab-toolbox-master/python_attention_pipeline/data/eeg_tensor.npy'
LABELS_PATH = '/data/home1/rishavs/atul/cocoha-matlab-toolbox-master/cocoha-matlab-toolbox-master/python_attention_pipeline/data/labels.npy'
AUDIO_DIR = '/data/home1/rishavs/atul/cocoha-matlab-toolbox-master/cocoha-matlab-toolbox-master/python_attention_pipeline/data/aligned_audio/'

# Parameters
subject_idx = 0  # Change as needed
trial_idx = 0    # Change as needed
fs_eeg = 32      # Hz, from data_statistics.txt

# Load EEG and labels
eeg = np.load(EEG_PATH, mmap_mode='r')  # shape: (subjects, trials, time, channels)
labels = np.load(LABELS_PATH)

# Get EEG for this subject/trial
eeg_trial = eeg[subject_idx, trial_idx]  # shape: (time, channels)
label = labels[subject_idx, trial_idx]
print(f'EEG shape (time, channels): {eeg_trial.shape}')
print(f'Label: {label}')

# EEG duration
n_eeg_samples = eeg_trial.shape[0]
dur_eeg = n_eeg_samples / fs_eeg
print(f'EEG duration: {dur_eeg:.2f} seconds')

# Find corresponding audio files (assuming S{subject+1}_trial{trial+1}_mixture_0/1.wav)
subj_str = f'S{subject_idx+1}'
trial_str = f'trial{trial_idx+1}'
audio0_path = os.path.join(AUDIO_DIR, f'{subj_str}_{trial_str}_mixture_0_aligned.wav')
audio1_path = os.path.join(AUDIO_DIR, f'{subj_str}_{trial_str}_mixture_1_aligned.wav')

# Load audio files
def load_audio_info(path):
    if not os.path.exists(path):
        print(f'Audio file not found: {path}')
        return None, None, None
    audio, fs_audio = sf.read(path)
    print(f'Loaded {path}: {len(audio)/fs_audio:.2f} s, fs={fs_audio}, shape: {audio.shape}')
    dur_audio = len(audio) / fs_audio
    return audio, fs_audio, dur_audio

audio0, fs_audio0, dur_audio0 = load_audio_info(audio0_path)
audio1, fs_audio1, dur_audio1 = load_audio_info(audio1_path)

print(f'Audio 0: {audio0_path}, duration: {dur_audio0:.2f} s, fs: {fs_audio0}')
print(f'Audio 1: {audio1_path}, duration: {dur_audio1:.2f} s, fs: {fs_audio1}')

# Check synchronization
if dur_audio0 is not None and abs(dur_audio0 - dur_eeg) > 0.1:
    print('WARNING: EEG and audio 0 durations do not match!')
if dur_audio1 is not None and abs(dur_audio1 - dur_eeg) > 0.1:
    print('WARNING: EEG and audio 1 durations do not match!')

# Optional: plot a short segment
if audio0 is not None:
    plt.figure(figsize=(10, 4))
    t_eeg = np.arange(n_eeg_samples) / fs_eeg
    t_audio = np.arange(len(audio0)) / fs_audio0
    plt.plot(t_eeg[:fs_eeg*5], eeg_trial[:fs_eeg*5, 0], label='EEG (ch 0)')
    plt.plot(t_audio[:fs_audio0*5], audio0[:fs_audio0*5], label='Audio 0', alpha=0.7)
    plt.title('First 5 seconds: EEG (ch 0) and Audio 0')
    plt.legend()
    plt.xlabel('Time (s)')
    plt.tight_layout()
    plt.show() 