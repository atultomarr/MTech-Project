import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'python_attention_pipeline/scripts/paper_implementation'))

import numpy as np
from eeg_filterbank import eeg_modulation_filterbank, get_window_lengths_samples
from tqdm import tqdm

# Load original EEG tensor
in_path = '/data/home1/rishavs/atul/cocoha-matlab-toolbox-master/cocoha-matlab-toolbox-master/python_attention_pipeline/data/eeg_tensor.npy'
out_path = '/data/home1/rishavs/atul/cocoha-matlab-toolbox-master/cocoha-matlab-toolbox-master/python_attention_pipeline/data/eeg_tensor_filterbank.npy'
eeg_tensor = np.load(in_path)  # (subjects, trials, time, channels)
fs = 32
n_bands = 10
window_lengths_samples = get_window_lengths_samples(fs, n_bands)

subjects, trials, timepoints, channels = eeg_tensor.shape
filtered_eeg_tensor = [] 

for subj in tqdm(range(subjects), desc='Subjects'):
    subj_trials = []
    for trial in tqdm(range(trials), desc=f'Subject {subj+1}', leave=False):
        trial_eeg = eeg_tensor[subj, trial]  # (time, channels)
        filtered = eeg_modulation_filterbank(trial_eeg, fs, window_lengths_samples)  # (time, channels * bands)
        subj_trials.append(filtered)
    filtered_eeg_tensor.append(subj_trials)

filtered_eeg_tensor = np.array(filtered_eeg_tensor)
print('Filtered EEG tensor shape:', filtered_eeg_tensor.shape)
np.save(out_path, filtered_eeg_tensor)
print(f'Saved filtered EEG tensor to {out_path}') 