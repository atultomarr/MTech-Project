import numpy as np
import os

def check_envelopes(files_to_check):
    ENVELOPE_DIR = 'python_attention_pipeline/data/paper_replication/audio_envelopes/'
    for fname in files_to_check:
        path = os.path.join(ENVELOPE_DIR, fname)
        if os.path.exists(path):
            env = np.load(path)
            n_nan = np.isnan(env).sum()
            arr_min = np.nanmin(env)
            arr_max = np.nanmax(env)
            arr_mean = np.nanmean(env)
            is_const = np.allclose(env, env.flat[0])
            is_zero = np.all(env == 0)
            print(f'{fname}: shape={env.shape}, NaNs={n_nan}, min={arr_min:.4f}, max={arr_max:.4f}, mean={arr_mean:.4f}, constant={is_const}, all_zero={is_zero}')
        else:
            print(f'{fname}: file not found')

if __name__ == '__main__':
    files = [
        'S1_trial2_mixture_0.npy',
        'S1_trial2_mixture_1.npy',
        'S1_trial4_mixture_0.npy',
        'S1_trial4_mixture_1.npy',
    ]
    check_envelopes(files) 