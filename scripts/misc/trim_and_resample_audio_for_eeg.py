import numpy as np
import soundfile as sf
import os
from scipy.signal import resample

# Parameters
subject_idx = 0  # Change as needed
trial_idx = 0    # Change as needed
fs_eeg = 32      # Hz
n_eeg_samples = 3968  # From EEG shape
trim_duration = n_eeg_samples / fs_eeg  # 124 seconds

AUDIO_DIR = '/data/home1/rishavs/atul/cocoha-matlab-toolbox-master/cocoha-matlab-toolbox-master/python_attention_pipeline/data/separated/'
OUTPUT_DIR = '/data/home1/rishavs/atul/cocoha-matlab-toolbox-master/cocoha-matlab-toolbox-master/python_attention_pipeline/data/aligned_audio/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

subj_str = f'S{subject_idx+1}'
trial_str = f'trial{trial_idx+1}'
audio0_path = os.path.join(AUDIO_DIR, f'{subj_str}_{trial_str}_mixture_0.wav')
audio1_path = os.path.join(AUDIO_DIR, f'{subj_str}_{trial_str}_mixture_1.wav')

# Helper to trim and resample
def trim_and_resample(audio, fs_audio, target_len, target_fs):
    # Trim
    n_trim = int(target_len * fs_audio)
    audio_trimmed = audio[:n_trim]
    # Resample
    n_target = int(target_len * target_fs)
    audio_resampled = resample(audio_trimmed, n_target)
    return audio_resampled

for idx, audio_path in enumerate([audio0_path, audio1_path]):
    if not os.path.exists(audio_path):
        print(f'Audio file not found: {audio_path}')
        continue
    audio, fs_audio = sf.read(audio_path)
    print(f'Loaded {audio_path}: {len(audio)/fs_audio:.2f} s, fs={fs_audio}')
    audio_aligned = trim_and_resample(audio, fs_audio, trim_duration, fs_eeg)
    print(f'After trim/resample: {len(audio_aligned)/fs_eeg:.2f} s, fs={fs_eeg}')
    out_path = os.path.join(OUTPUT_DIR, f'{subj_str}_{trial_str}_mixture_{idx}_aligned.wav')
    sf.write(out_path, audio_aligned, fs_eeg)
    print(f'Saved aligned audio to {out_path}') 