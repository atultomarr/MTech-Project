import numpy as np
import os
import pickle
from scipy.stats import zscore, pearsonr

# Parameters
EXPERIMENTS = {
    'exp1': {'window_sec': 8.0, 'stride_sec': 0.25},
    'exp2': {'window_sec': 6.5, 'stride_sec': 0.25},
}
CCA_OUTPUT_DIR = '/data/home1/rishavs/atul/cocoha-matlab-toolbox-master/cocoha-matlab-toolbox-master/python_attention_pipeline/data/paper_replication/cca_outputs/'
SAVE_DIR = '/data/home1/rishavs/atul/cocoha-matlab-toolbox-master/cocoha-matlab-toolbox-master/python_attention_pipeline/data/paper_replication/sliding_corr_outputs/'
os.makedirs(SAVE_DIR, exist_ok=True)

N_SUBJECTS = 16
FS = 32  # Sampling rate (Hz)

# Helper: sliding window correlation
def sliding_corr(x, y, window_size, stride):
    n = len(x)
    win_samples = int(window_size * FS)
    stride_samples = int(stride * FS)
    corrs = []
    idxs = []
    for start in range(0, n - win_samples + 1, stride_samples):
        end = start + win_samples
        if end > n:
            break
        r = pearsonr(x[start:end], y[start:end])[0]
        corrs.append(r)
        idxs.append(start)
    return np.array(corrs), np.array(idxs)

def process_subject(subj):
    print(f'Processing subject {subj+1}')
    # Load train and test CCA outputs
    X_train_cca = np.load(os.path.join(CCA_OUTPUT_DIR, f'subj{subj+1}_X_train_cca.npy'))
    Y_train_cca = np.load(os.path.join(CCA_OUTPUT_DIR, f'subj{subj+1}_Y_train_cca.npy'))
    with open(os.path.join(CCA_OUTPUT_DIR, f'subj{subj+1}_X_test_cca.pkl'), 'rb') as f:
        X_test_cca = pickle.load(f)
    with open(os.path.join(CCA_OUTPUT_DIR, f'subj{subj+1}_Y_test_cca.pkl'), 'rb') as f:
        Y_test_cca = pickle.load(f)
    
    # Get train/test split indices
    train_idx = np.load(os.path.join(CCA_OUTPUT_DIR, f'subj{subj+1}_train_idx.npy'))
    test_idx = np.load(os.path.join(CCA_OUTPUT_DIR, f'subj{subj+1}_test_idx.npy'))
    
    # For each experiment
    for exp_name, params in EXPERIMENTS.items():
        window_sec = params['window_sec']
        stride_sec = params['stride_sec']
        
        # Training set windowed correlations
        if subj == 0 and exp_name == 'exp1':
            print('Inspecting CCA output lengths for first few training trials:')
            for i in range(min(5, len(X_train_cca))):
                print(f'Trial {i}: X_train_cca[{i}].shape = {X_train_cca[i].shape}, Y_train_cca[{i}].shape = {Y_train_cca[i].shape}')
        train_corrs_list = []
        for i in range(len(train_idx)):
            x = X_train_cca[i]
            y = Y_train_cca[i]
            x1d = x if x.ndim == 1 else x[:, 0]
            y1d = y if y.ndim == 1 else y[:, 0]
            corrs, idxs = sliding_corr(x1d, y1d, window_sec, stride_sec)
            train_corrs_list.append({'corrs': corrs, 'idxs': idxs})
        
        # Compute z-score stats from training set
        all_train_corrs = np.concatenate([d['corrs'] for d in train_corrs_list])
        train_mean = np.mean(all_train_corrs)
        train_std = np.std(all_train_corrs)
        
        # Z-score training correlations
        for d in train_corrs_list:
            d['corrs_z'] = (d['corrs'] - train_mean) / (train_std + 1e-8)
        
        # Test set windowed correlations
        test_corrs_list = []
        for i in range(len(test_idx)):
            x = X_test_cca[i]
            y = Y_test_cca[i]
            x1d = x if x.ndim == 1 else x[:, 0]
            y1d = y if y.ndim == 1 else y[:, 0]
            corrs, idxs = sliding_corr(x1d, y1d, window_sec, stride_sec)
            corrs_z = (corrs - train_mean) / (train_std + 1e-8)
            test_corrs_list.append({'corrs': corrs, 'corrs_z': corrs_z, 'idxs': idxs})
        
        # Save both train and test correlations
        with open(os.path.join(SAVE_DIR, f'subj{subj+1}_{exp_name}_train_corrs.pkl'), 'wb') as f:
            pickle.dump(train_corrs_list, f)
        with open(os.path.join(SAVE_DIR, f'subj{subj+1}_{exp_name}_test_corrs.pkl'), 'wb') as f:
            pickle.dump(test_corrs_list, f)
        
        print(f'Saved {exp_name} windowed correlations for subject {subj+1}')
        
        # Print summary statistics
        all_train_corrs_z = np.concatenate([d['corrs_z'] for d in train_corrs_list]) if train_corrs_list else np.array([])
        all_test_corrs_z = np.concatenate([d['corrs_z'] for d in test_corrs_list]) if test_corrs_list else np.array([])

        print(f"{exp_name} training correlations for subject {subj+1}:")
        if all_train_corrs_z.size > 0:
            print(f"  Mean: {np.mean(all_train_corrs_z):.4f}, Std: {np.std(all_train_corrs_z):.4f}, Min: {np.min(all_train_corrs_z):.4f}, Max: {np.max(all_train_corrs_z):.4f}")
        else:
            print("  No training correlations for this subject/experiment.")
        print(f"{exp_name} test correlations for subject {subj+1}:")
        if all_test_corrs_z.size > 0:
            print(f"  Mean: {np.mean(all_test_corrs_z):.4f}, Std: {np.std(all_test_corrs_z):.4f}, Min: {np.min(all_test_corrs_z):.4f}, Max: {np.max(all_test_corrs_z):.4f}")
        else:
            print("  No test correlations for this subject/experiment.")

def main():
    for subj in range(N_SUBJECTS):
        process_subject(subj)

if __name__ == '__main__':
    main() 