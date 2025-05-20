import numpy as np
import pandas as pd
import soundfile as sf
import matplotlib.pyplot as plt

# File paths (edit as needed)
STATE_CSV = 'state_tracker_output_subject1_trial1_stream1.csv'
OUTPUT_WAV = 'amplified_audio_subject1_trial1.wav'
ORIG0_WAV = '/data/home1/rishavs/atul/cocoha-matlab-toolbox-master/cocoha-matlab-toolbox-master/python_attention_pipeline/data/separated/S1_trial1_mixture_0.wav'
ORIG1_WAV = '/data/home1/rishavs/atul/cocoha-matlab-toolbox-master/cocoha-matlab-toolbox-master/python_attention_pipeline/data/separated/S1_trial1_mixture_1.wav'

# Load state trajectory
state = pd.read_csv(STATE_CSV)['state'].values

# Load audio
output, fs = sf.read(OUTPUT_WAV)
orig0, _ = sf.read(ORIG0_WAV)
orig1, _ = sf.read(ORIG1_WAV)

# Print RMS values
print('RMS values:')
print('  Amplified Output:', np.sqrt(np.mean(output**2)))
print('  Original Stream 0:', np.sqrt(np.mean(orig0**2)))
print('  Original Stream 1:', np.sqrt(np.mean(orig1**2)))

# Plot state trajectory
plt.figure(figsize=(12, 4))
plt.plot(state, label='State (attention)', color='C2')
plt.title('State Trajectory')
plt.xlabel('Window Index')
plt.ylabel('State')
plt.legend()
plt.tight_layout()
plt.savefig('state_trajectory.png')
plt.close()

# Plot waveforms (first 10 seconds for clarity)
duration_sec = 10
samples_to_plot = int(fs * duration_sec)
plt.figure(figsize=(12, 6))
plt.plot(output[:samples_to_plot], label='Amplified Output', alpha=0.8)
plt.plot(orig0[:samples_to_plot], label='Original Stream 0', alpha=0.5)
plt.plot(orig1[:samples_to_plot], label='Original Stream 1', alpha=0.5)
plt.title('Waveforms (First 10 seconds)')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.legend()
plt.tight_layout()
plt.savefig('waveforms.png')
plt.close()

# Overlay state trajectory (upsampled to audio rate)
from scipy.signal import resample
state_upsampled = resample(state, len(output))
plt.figure(figsize=(12, 6))
plt.plot(output[:samples_to_plot], label='Amplified Output', alpha=0.8)
plt.plot(state_upsampled[:samples_to_plot], label='Upsampled State', color='C2', alpha=0.7)
plt.title('Output and Upsampled State (First 10 seconds)')
plt.xlabel('Sample Index')
plt.legend()
plt.tight_layout()
plt.savefig('output_and_state_overlay.png')
plt.close() 