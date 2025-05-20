import numpy as np
import os

def check_raw_features(subj=1, trials=[1, 3]):
    EEG_PATH = 'python_attention_pipeline/data/eeg_tensor.npy'
    AUDIO_ENV_DIR = 'python_attention_pipeline/data/paper_replication/audio_envelopes/'
    eeg_tensor = np.load(EEG_PATH)
    for trial in trials:
        print(f'\nSubject {subj}, Trial {trial+1}:')
        eeg = eeg_tensor[subj-1, trial]  # (time, channels)
        subj_str = f'S{subj}'
        trial_str = f'trial{trial+1}'
        env0_path = os.path.join(AUDIO_ENV_DIR, f'{subj_str}_{trial_str}_mixture_0.npy')
        env1_path = os.path.join(AUDIO_ENV_DIR, f'{subj_str}_{trial_str}_mixture_1.npy')
        env0 = np.load(env0_path).T
        env1 = np.load(env1_path).T
        audio_env = np.concatenate([env0, env1], axis=1)
        min_len = min(eeg.shape[0], audio_env.shape[0])
        eeg = eeg[:min_len]
        audio_env = audio_env[:min_len]
        for name, arr in [('EEG', eeg), ('Audio', audio_env)]:
            n_nan = np.isnan(arr).sum()
            arr_min = np.nanmin(arr)
            arr_max = np.nanmax(arr)
            arr_mean = np.nanmean(arr)
            is_const = np.allclose(arr, arr[0])
            is_zero = np.all(arr == 0)
            print(f'  {name}: shape={arr.shape}, NaNs={n_nan}, min={arr_min:.4f}, max={arr_max:.4f}, mean={arr_mean:.4f}, constant={is_const}, all_zero={is_zero}')

if __name__ == '__main__':
    check_raw_features(1, [1, 3]) 