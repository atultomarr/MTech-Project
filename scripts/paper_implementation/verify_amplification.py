import numpy as np
import pandas as pd
import soundfile as sf
import os
from scipy.signal import get_window
import matplotlib.pyplot as plt

# Load state tracker results
STATE_CSV = '/data/home1/rishavs/atul/cocoha-matlab-toolbox-master/cocoha-matlab-toolbox-master/python_attention_pipeline/data/paper_replication/state_tracker_results/state_tracker_results.csv'
state_df = pd.read_csv(STATE_CSV)

# Parameters
AUDIO_DIR = '/data/home1/rishavs/atul/cocoha-matlab-toolbox-master/cocoha-matlab-toolbox-master/python_attention_pipeline/data/separated'
AMPLIFIED_DIR = '/data/home1/rishavs/atul/cocoha-matlab-toolbox-master/cocoha-matlab-toolbox-master/python_attention_pipeline/data/paper_replication/amplified_audio'
WINDOW_SEC = 8.0
STRIDE_SEC = 0.25
AUDIO_FS = 8000

# Create output directory for plots
PLOT_DIR = '/data/home1/rishavs/atul/cocoha-matlab-toolbox-master/cocoha-matlab-toolbox-master/python_attention_pipeline/data/paper_replication/amplification_plots'
os.makedirs(PLOT_DIR, exist_ok=True)

def verify_amplification(subject, trial):
    # Get state trajectory
    mask = (state_df['subject'] == subject) & (state_df['trial'] == trial)
    state = state_df.loc[mask, 'state'].values
    
    # Load original audio
    env0_path = os.path.join(AUDIO_DIR, f'S{subject}_trial{trial}_mixture_0.wav')
    env1_path = os.path.join(AUDIO_DIR, f'S{subject}_trial{trial}_mixture_1.wav')
    audio0, _ = sf.read(env0_path)
    audio1, _ = sf.read(env1_path)
    
    # Load amplified audio
    amp_path = os.path.join(AMPLIFIED_DIR, f'amplified_audio_subject{subject}_trial{trial}.wav')
    amplified, _ = sf.read(amp_path)
    
    # Calculate window parameters
    window_len = int(WINDOW_SEC * AUDIO_FS)
    stride_len = int(STRIDE_SEC * AUDIO_FS)
    
    # Calculate expected amplification
    expected = np.zeros_like(amplified)
    for i, s in enumerate(state):
        start = i * stride_len
        end = start + window_len
        if end > len(expected):
            break
        gain0 = (1 - s) / 2  # Unattended
        gain1 = (1 + s) / 2  # Attended
        window_func = get_window('hann', window_len)
        expected[start:end] += (audio0[start:end] * gain0 + audio1[start:end] * gain1) * window_func
    
    # Normalize expected output
    expected = expected / np.max(np.abs(expected)) * 0.99
    
    # Calculate correlation between expected and actual
    correlation = np.corrcoef(expected, amplified)[0, 1]
    
    # Create figure with detailed annotations
    plt.figure(figsize=(15, 12))
    
    # Plot state trajectory
    plt.subplot(3, 1, 1)
    plt.plot(state, 'b-', label='State Value')
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Neutral Point')
    plt.title(f'State Trajectory (Subject {subject}, Trial {trial})', fontsize=12)
    plt.xlabel('Window Number', fontsize=10)
    plt.ylabel('State Value (-1 to 1)', fontsize=10)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add annotations for state values
    plt.annotate('State > 0: Attending to Stream 1', xy=(0.02, 0.95), xycoords='axes fraction', 
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    plt.annotate('State < 0: Attending to Stream 0', xy=(0.02, 0.05), xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    # Plot original audio
    plt.subplot(3, 1, 2)
    plt.plot(audio0, label='Stream 0', alpha=0.5, color='blue')
    plt.plot(audio1, label='Stream 1', alpha=0.5, color='red')
    plt.title('Original Audio Streams', fontsize=12)
    plt.xlabel('Sample Number', fontsize=10)
    plt.ylabel('Amplitude', fontsize=10)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot amplified vs expected
    plt.subplot(3, 1, 3)
    plt.plot(amplified, label='Actual Amplified', alpha=0.5, color='green')
    plt.plot(expected, label='Expected Amplified', alpha=0.5, color='purple')
    plt.title(f'Amplified Audio (Correlation: {correlation:.3f})', fontsize=12)
    plt.xlabel('Sample Number', fontsize=10)
    plt.ylabel('Amplitude', fontsize=10)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add explanation of amplification
    plt.annotate('Amplification Process:\n' +
                '1. State > 0: Stream 1 amplified, Stream 0 attenuated\n' +
                '2. State < 0: Stream 0 amplified, Stream 1 attenuated\n' +
                '3. State = 0: Both streams at equal volume',
                xy=(0.02, 0.95), xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f'amplification_verification_subject{subject}_trial{trial}_detailed.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return correlation

# Verify just one example for detailed explanation
subject = 1
trial = 1
try:
    correlation = verify_amplification(subject, trial)
    print(f'Subject {subject}, Trial {trial}: Correlation = {correlation:.3f}')
except Exception as e:
    print(f'Error processing Subject {subject}, Trial {trial}: {str(e)}') 