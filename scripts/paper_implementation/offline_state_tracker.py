import numpy as np
import pandas as pd
import os
from tqdm import tqdm

# Ported state tracker logic from OVStateTracker.py

def offline_state_tracker(probabilities, epoch_len=8.0, sampling_rate=128, lmb=0.1, exponent=1.0, nclass=2):
    """
    Parameters:
        probabilities: np.ndarray, classifier probabilities for attended class (shape: [n_windows])
        epoch_len: float, epoch/window length in seconds
        sampling_rate: float, sampling rate (Hz)
        lmb: float, decay parameter (lambda)
        exponent: float, exponent for nonlinearity
        nclass: int, number of classes (default 2)
    Returns:
        state: np.ndarray, smoothed state trajectory (between -1 and 1)
    """
    state = np.zeros_like(probabilities)
    prev_state = 0.0
    for i, p in enumerate(probabilities):
        # Information transfer rate (ITR) in bits per epoch
        if p == 0 or p == 1:
            itr = 1.0  # avoid log(0)
        else:
            itr = (np.log2(nclass) + p * np.log2(p) + (1 - p) * np.log2((1 - p) / (nclass - 1)))
        # State update
        delta = (itr * np.sign(p - 0.5) - lmb * np.abs(prev_state) ** exponent * np.sign(prev_state))
        delta *= 2 / (sampling_rate * epoch_len)
        new_state = prev_state + delta
        # Bound state between -1 and 1
        new_state = max(min(new_state, 1.0), -1.0)
        state[i] = new_state
        prev_state = new_state
    return state

if __name__ == "__main__":
    # Load the decoding results
    csv_path = "/data/home1/rishavs/atul/cocoha-matlab-toolbox-master/cocoha-matlab-toolbox-master/python_attention_pipeline/data/paper_replication/decoding_results/cca_attention_decoding_results_full.csv"
    df = pd.read_csv(csv_path)
    
    # Create output directory if it doesn't exist
    output_dir = "../../data/paper_replication/state_tracker_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Get unique combinations of subject, trial, and experiment
    unique_combinations = df[['subject', 'trial', 'experiment']].drop_duplicates()
    
    # Process each combination
    all_results = []
    for _, row in tqdm(unique_combinations.iterrows(), desc="Processing combinations"):
        subject = row['subject']
        trial = row['trial']
        experiment = row['experiment']
        
        # Get probabilities for this combination
        mask = (df['subject'] == subject) & (df['trial'] == trial) & (df['experiment'] == experiment)
        probs = df.loc[mask, 'prob_1'].values
        
        if len(probs) > 0:
            # Apply state tracker
            state = offline_state_tracker(probs, epoch_len=8.0, sampling_rate=128, lmb=0.1, exponent=1.0)
            
            # Store results
            for i, (prob, state_val) in enumerate(zip(probs, state)):
                all_results.append({
                    'subject': subject,
                    'trial': trial,
                    'experiment': experiment,
                    'window': i,
                    'probability': prob,
                    'state': state_val
                })
    
    # Convert to DataFrame and save
    results_df = pd.DataFrame(all_results)
    output_path = os.path.join(output_dir, 'state_tracker_results.csv')
    results_df.to_csv(output_path, index=False)
    print(f'State tracker results saved to {output_path}') 