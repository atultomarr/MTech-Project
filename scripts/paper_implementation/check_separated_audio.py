import numpy as np
import os
import soundfile as sf

def check_separated_audio(files_to_check):
    AUDIO_DIR = 'python_attention_pipeline/data/separated/'
    for fname in files_to_check:
        path = os.path.join(AUDIO_DIR, fname)
        if os.path.exists(path):
            audio, sr = sf.read(path)
            n_nan = np.isnan(audio).sum()
            arr_min = np.nanmin(audio)
            arr_max = np.nanmax(audio)
            arr_mean = np.nanmean(audio)
            is_const = np.allclose(audio, audio[0])
            is_zero = np.all(audio == 0)
            print(f'{fname}: shape={audio.shape}, NaNs={n_nan}, min={arr_min:.4f}, max={arr_max:.4f}, mean={arr_mean:.4f}, constant={is_const}, all_zero={is_zero}')
        else:
            print(f'{fname}: file not found')

if __name__ == '__main__':
    files = [
        'S1_trial2_mixture_0.wav',
        'S1_trial2_mixture_1.wav',
        'S1_trial4_mixture_0.wav',
        'S1_trial4_mixture_1.wav',
    ]
    check_separated_audio(files) 