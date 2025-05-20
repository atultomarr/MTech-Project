import numpy as np
import pandas as pd
import soundfile as sf
import os
from scipy.signal import get_window
from tqdm import tqdm

# Parameters
AUDIO_DIR = '/data/home1/rishavs/atul/cocoha-matlab-toolbox-master/cocoha-matlab-toolbox-master/python_attention_pipeline/data/separated'
STATE_CSV = '/data/home1/rishavs/atul/cocoha-matlab-toolbox-master/cocoha-matlab-toolbox-master/python_attention_pipeline/data/paper_replication/state_tracker_results/state_tracker_results.csv'
OUTPUT_DIR = '/data/home1/rishavs/atul/cocoha-matlab-toolbox-master/cocoha-matlab-toolbox-master/python_attention_pipeline/data/paper_replication/amplified_audio'
WINDOW_SEC = 8.0
STRIDE_SEC = 0.25
FS = 128  # Downsampled rate used in your pipeline
AUDIO_FS = 8000  # Actual audio file sample rate

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load state trajectories
state_df = pd.read_csv(STATE_CSV)

# Get unique combinations of subject and trial
unique_combinations = state_df[['subject', 'trial']].drop_duplicates()

# Process each combination
for _, row in tqdm(unique_combinations.iterrows(), desc="Processing subjects and trials"):
    subject = row['subject']
    trial = row['trial']
    
    # Get state trajectory for this subject and trial
    mask = (state_df['subject'] == subject) & (state_df['trial'] == trial)
    state = state_df.loc[mask, 'state'].values
    
    # Load audio streams
    env0_path = os.path.join(AUDIO_DIR, f'S{subject}_trial{trial}_mixture_0.wav')
    env1_path = os.path.join(AUDIO_DIR, f'S{subject}_trial{trial}_mixture_1.wav')
    
    try:
        audio0, fs0 = sf.read(env0_path)
        audio1, fs1 = sf.read(env1_path)
        assert fs0 == fs1 == AUDIO_FS, f'Audio sample rates do not match for subject {subject}, trial {trial}!'
        
        min_len = min(len(audio0), len(audio1))
        audio0 = audio0[:min_len]
        audio1 = audio1[:min_len]

        # Calculate window/stride in samples (audio rate)
        window_len = int(WINDOW_SEC * AUDIO_FS)
        stride_len = int(STRIDE_SEC * AUDIO_FS)

        # Prepare output
        output = np.zeros(min_len)
        window_count = len(state)

        for i in range(window_count):
            start = i * stride_len
            end = start + window_len
            if end > min_len:
                break
            # Compute gain for each stream
            gain0 = (1 - state[i]) / 2  # Unattended
            gain1 = (1 + state[i]) / 2  # Attended
            # Apply gain to window
            win0 = audio0[start:end] * gain0
            win1 = audio1[start:end] * gain1
            # Overlap-add (Hann window for smoothness)
            window_func = get_window('hann', window_len)
            output[start:end] += (win0 + win1) * window_func

        # Normalize output to prevent clipping
        output = output / np.max(np.abs(output)) * 0.99

        # Save output
        output_path = os.path.join(OUTPUT_DIR, f'amplified_audio_subject{subject}_trial{trial}.wav')
        sf.write(output_path, output, AUDIO_FS)
        print(f'Amplified audio saved to {output_path}')
        
    except Exception as e:
        print(f'Error processing subject {subject}, trial {trial}: {str(e)}')
        continue 