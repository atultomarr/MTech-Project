import numpy as np
import soundfile as sf
from scipy.signal import hilbert
from sklearn.cross_decomposition import CCA
from scipy.stats import zscore
import os
import csv
from tqdm import tqdm
from collections import Counter

# Parameters
window_len = 256  # 8 seconds at 32 Hz
step_size = 128   # 4 seconds (50% overlap)

# Paths
EEG_PATH = 'python_attention_pipeline/data/eeg_tensor.npy'
LABELS_PATH = 'python_attention_pipeline/data/labels.npy'
AUDIO_DIR = 'python_attention_pipeline/data/aligned_audio/'
CSV_PATH = 'cca_attention_windowed_results.csv'

# Load EEG and labels
eeg = np.load(EEG_PATH, mmap_mode='r')  # (subjects, trials, time, channels)
labels = np.load(LABELS_PATH)  # (subjects, trials)

n_subjects, n_trials, n_time, n_channels = eeg.shape
fs = 32

results = []
per_window_correct = 0
per_window_total = 0
per_trial_majority = []

for subj in tqdm(range(n_subjects), desc='Subjects'):
    for trial in tqdm(range(n_trials), desc=f'Trials (Subject {subj})', leave=False):
        try:
            eeg_trial = eeg[subj, trial]
            stds = np.std(eeg_trial, axis=0)
            nonconst_mask = stds > 0
            eeg_trial_nonconst = eeg_trial[:, nonconst_mask]
            if eeg_trial_nonconst.shape[1] == 0:
                continue
            eeg_z = zscore(eeg_trial_nonconst, axis=0)
            valid_mask = ~np.isnan(eeg_z).any(axis=1)
            eeg_z_valid = eeg_z[valid_mask]
            if eeg_z_valid.shape[0] < window_len:
                continue
            subj_str = f'S{subj+1}'
            trial_str = f'trial{trial+1}'
            audio0_path = os.path.join(AUDIO_DIR, f'{subj_str}_{trial_str}_mixture_0_aligned.wav')
            audio1_path = os.path.join(AUDIO_DIR, f'{subj_str}_{trial_str}_mixture_1_aligned.wav')
            if not (os.path.exists(audio0_path) and os.path.exists(audio1_path)):
                continue
            audio0, fs0 = sf.read(audio0_path)
            audio1, fs1 = sf.read(audio1_path)
            if fs0 != fs or fs1 != fs or len(audio0) != n_time or len(audio1) != n_time:
                continue
            env0 = zscore(np.abs(hilbert(audio0)))[valid_mask]
            env1 = zscore(np.abs(hilbert(audio1)))[valid_mask]
            n_valid = eeg_z_valid.shape[0]
            window_preds = []
            for start in range(0, n_valid - window_len + 1, step_size):
                stop = start + window_len
                eeg_win = eeg_z_valid[start:stop]
                env0_win = env0[start:stop]
                env1_win = env1[start:stop]
                cca = CCA(n_components=1)
                cca.fit(eeg_win, env0_win.reshape(-1, 1))
                X_c0, Y_c0 = cca.transform(eeg_win, env0_win.reshape(-1, 1))
                corr0 = np.corrcoef(X_c0[:, 0], Y_c0[:, 0])[0, 1]
                cca.fit(eeg_win, env1_win.reshape(-1, 1))
                X_c1, Y_c1 = cca.transform(eeg_win, env1_win.reshape(-1, 1))
                corr1 = np.corrcoef(X_c1[:, 0], Y_c1[:, 0])[0, 1]
                predicted = int(np.argmax([corr0, corr1]))
                true_label = int(labels[subj, trial])
                results.append([subj, trial, start, corr0, corr1, predicted, true_label])
                window_preds.append(predicted)
                per_window_total += 1
                if predicted == true_label:
                    per_window_correct += 1
            if window_preds:
                maj = Counter(window_preds).most_common(1)[0][0]
                per_trial_majority.append((maj, true_label))
        except Exception as e:
            continue

# Save to CSV
with open(CSV_PATH, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['subject', 'trial', 'window_start', 'corr0', 'corr1', 'predicted', 'true_label'])
    writer.writerows(results)

# Compute per-window accuracy
window_acc = per_window_correct / per_window_total if per_window_total > 0 else 0
print(f'Per-window accuracy: {window_acc*100:.2f}% ({per_window_correct}/{per_window_total})')

# Compute per-trial (majority vote) accuracy
if per_trial_majority:
    per_trial_acc = np.mean([maj == true for maj, true in per_trial_majority])
    print(f'Per-trial (majority vote) accuracy: {per_trial_acc*100:.2f}% ({sum(maj == true for maj, true in per_trial_majority)}/{len(per_trial_majority)})')
else:
    print('No valid trials for per-trial accuracy.')
print(f'Results saved to {CSV_PATH}') 