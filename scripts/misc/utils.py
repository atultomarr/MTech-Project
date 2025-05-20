import os
import numpy as np
import soundfile as sf
import json

# Path to mapping file and audio directory
MAPPING_PATH = 'python_attention_pipeline/data/trial_audio_mapping.json'
AUDIO_DIR = 'python_attention_pipeline/data/Stimuli dry'
OUTPUT_MIX_DIR = 'python_attention_pipeline/data/mixtures'

# Ensure output directory exists
os.makedirs(OUTPUT_MIX_DIR, exist_ok=True)

def to_dry_filename(filename):
    """Convert any .wav filename to its _dry.wav version."""
    if filename.endswith('_dry.wav'):
        return filename
    base = filename.replace('_hrtf.wav', '').replace('.wav', '')
    if base.startswith('rep_'):
        # e.g., rep_part1_track1_hrtf.wav -> rep_part1_track1_dry.wav
        return base + '_dry.wav'
    else:
        # e.g., part1_track1_hrtf.wav -> part1_track1_dry.wav
        return base + '_dry.wav'

def mix_audio_for_trial(left_path, right_path, output_path):
    """Mix left and right audio and save the mixture."""
    left_audio, sr_left = sf.read(left_path)
    right_audio, sr_right = sf.read(right_path)
    assert sr_left == sr_right, 'Sampling rates must match!'
    min_len = min(len(left_audio), len(right_audio))
    left_audio = left_audio[:min_len]
    right_audio = right_audio[:min_len]
    mixture = left_audio + right_audio
    mixture = mixture / np.max(np.abs(mixture))
    sf.write(output_path, mixture, sr_left)
    print(f'Mixed audio saved to {output_path}')

def automate_mixing_from_mapping():
    """Automate mixing for all trials using the mapping file, always using _dry.wav files."""
    with open(MAPPING_PATH, 'r') as f:
        mapping = json.load(f)
    for entry in mapping:
        left_file = to_dry_filename(entry['left_audio'])
        right_file = to_dry_filename(entry['right_audio'])
        subject = entry['subject']
        trial_idx = entry['trial_idx']
        # Compose full paths
        left_path = os.path.join(AUDIO_DIR, left_file)
        right_path = os.path.join(AUDIO_DIR, right_file)
        # Output file name includes subject and trial index
        output_path = os.path.join(OUTPUT_MIX_DIR, f'{subject}_trial{trial_idx+1}_mixture.wav')
        # Only mix if both files exist
        if os.path.exists(left_path) and os.path.exists(right_path):
            mix_audio_for_trial(left_path, right_path, output_path)
        else:
            print(f'Warning: Missing file for {subject} trial {trial_idx+1}: {left_file}, {right_file}')

if __name__ == '__main__':
    automate_mixing_from_mapping()
