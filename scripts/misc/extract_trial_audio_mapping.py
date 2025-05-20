import os
import json
from scipy.io import loadmat

# Directory containing subject .mat files
MAT_DIR = 'python_attention_pipeline/data/mat_subjects'
# Output mapping file
OUTPUT_JSON = 'python_attention_pipeline/data/trial_audio_mapping.json'

# List all .mat files in the directory
mat_files = sorted([f for f in os.listdir(MAT_DIR) if f.endswith('.mat')])

trial_audio_mapping = []  # List of dicts: {subject, trial, left_audio, right_audio, repetition}

for mat_file in mat_files:
    subject_id = os.path.splitext(mat_file)[0]
    mat_path = os.path.join(MAT_DIR, mat_file)
    data = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    # The structure may be 'trials' or 'preproc_trials' depending on the file
    trials = None
    if 'preproc_trials' in data:
        trials = data['preproc_trials']
    elif 'trials' in data:
        trials = data['trials']
    else:
        raise ValueError(f'No trials found in {mat_file}')
    # trials is an array of structs
    for trial_idx, trial in enumerate(trials):
        # Each trial has a 'stimuli' field (cell array of 2 strings)
        left_audio = str(trial.stimuli[0])
        right_audio = str(trial.stimuli[1])
        repetition = bool(getattr(trial, 'repetition', False))
        trial_audio_mapping.append({
            'subject': subject_id,
            'trial_idx': int(trial_idx),
            'left_audio': left_audio,
            'right_audio': right_audio,
            'repetition': repetition
        })

# Save mapping as JSON
with open(OUTPUT_JSON, 'w') as f:
    json.dump(trial_audio_mapping, f, indent=2)

print(f'Trial-to-audio mapping saved to {OUTPUT_JSON}') 