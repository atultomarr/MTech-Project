import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import soundfile as sf
from scipy.signal import hilbert
import warnings
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from scipy.io import loadmat

# Parameters
EEG_PATH = '/data/home1/rishavs/atul/cocoha-matlab-toolbox-master/cocoha-matlab-toolbox-master/python_attention_pipeline/data/eeg_tensor_filterbank.npy'
AUDIO_DIR = '/data/home1/rishavs/atul/cocoha-matlab-toolbox-master/cocoha-matlab-toolbox-master/python_attention_pipeline/data/aligned_audio'
LABELS_PATH = '/data/home1/rishavs/atul/cocoha-matlab-toolbox-master/cocoha-matlab-toolbox-master/python_attention_pipeline/data/labels.npy'
OUTPUT_DIR = '/data/home1/rishavs/atul/cocoha-matlab-toolbox-master/cocoha-matlab-toolbox-master/python_attention_pipeline/data/paper_replication/decoding_results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Experiment settings
EXPERIMENTS = {
    'exp1': {'window_sec': 8.0, 'stride_sec': 1},  # Using 0.25s stride for better temporal resolution
    'exp2': {'window_sec': 6.5, 'stride_sec': 1},  # Using 0.25s stride for better temporal resolution
}
DOWNSAMPLED_FS = 128
N_PCA = 0.99  # keep 99% variance
N_CCA = 1     # number of CCA components

# Helper to load audio features for a subject/trial
def load_audio_features(subject_idx, trial_idx, n_trials):
    env0_path = os.path.join(AUDIO_DIR, f'S{subject_idx+1}_trial{trial_idx+1}_mixture_0_aligned.wav')
    env1_path = os.path.join(AUDIO_DIR, f'S{subject_idx+1}_trial{trial_idx+1}_mixture_1_aligned.wav')
    env0, fs = sf.read(env0_path)
    env1, fs = sf.read(env1_path)
    env0 = np.abs(hilbert(env0))
    env1 = np.abs(hilbert(env1))
    return env0, env1

# Sliding window generator
def sliding_windows(data, window_len, stride):
    """Create sliding windows from data"""
    n_windows = (len(data) - window_len) // stride + 1
    windows = []
    for i in range(n_windows):
        start = i * stride
        end = start + window_len
        windows.append(data[start:end])
    return np.array(windows)

def compute_window_correlations(eeg_win, env_win):
    # PCA for EEG and audio
    pca_eeg = PCA(n_components=N_PCA)
    pca_audio = PCA(n_components=N_PCA)
    eeg_win_pca = pca_eeg.fit_transform(eeg_win)
    env_win_pca = pca_audio.fit_transform(env_win.reshape(-1, 1))
    # CCA
    cca = CCA(n_components=N_CCA)
    eeg_win_cca, env_win_cca = cca.fit_transform(eeg_win_pca, env_win_pca)
    return np.corrcoef(eeg_win_cca[:, 0], env_win_cca[:, 0])[0, 1]

# Main decoding function
def run_decoding():
    print("Script started.")
    eeg_tensor = np.load(EEG_PATH, mmap_mode='r')
    labels = np.load(LABELS_PATH)
    print(f"EEG tensor shape: {eeg_tensor.shape}")
    print(f"Labels shape: {labels.shape}")
    n_subjects, n_trials, n_samples, n_channels = eeg_tensor.shape
    results = []

    # Main progress bar for experiments
    for exp_name, exp_params in tqdm(EXPERIMENTS.items(), desc="Experiments", position=0):
        print(f"\nProcessing experiment: {exp_name}")
        window_len = int(exp_params['window_sec'] * DOWNSAMPLED_FS)
        stride = int(exp_params['stride_sec'] * DOWNSAMPLED_FS)
        
        # Process all subjects
        for test_subj in [0, 1]:  # Test on first two subjects
            training_subjects = [i for i in range(n_subjects) if i not in [0, 1]]  # Use all subjects except test subjects
            print(f"\nProcessing test subject {test_subj + 1}")
            # Prepare training data
            X_train, y_train = [], []
            
            # Process training data
            print(f"\nProcessing {n_trials} training trials from {len(training_subjects)} subjects...")
            for subj in tqdm(training_subjects, desc="Training subjects", position=2, leave=False):
                for trial in tqdm(range(n_trials), desc=f"Subject {subj+1} trials", position=3, leave=False):
                    if np.isnan(eeg_tensor[subj, trial]).all():
                        print(f"Skipping NaN trial {trial+1} for subject {subj+1}")
                        continue
                    
                    eeg_feat = eeg_tensor[subj, trial]
                    env0, env1 = load_audio_features(subj, trial, n_trials)
                    attn = labels[subj, trial] - 1
                    min_len = min(eeg_feat.shape[0], env0.shape[0], env1.shape[0])
                    eeg_feat = eeg_feat[:min_len]
                    env0 = env0[:min_len]
                    env1 = env1[:min_len]
                    
                    # Process windows
                    n_windows = 0
                    for start in range(0, min_len - window_len + 1, stride):
                        end = start + window_len
                        eeg_win = eeg_feat[start:end]
                        env0_win = env0[start:end]
                        env1_win = env1[start:end]
                        corrs = [
                            compute_window_correlations(eeg_win, env0_win),
                            compute_window_correlations(eeg_win, env1_win)
                        ]
                        X_train.append(corrs)
                        y_train.append(attn)
                        n_windows += 1
                    print(f"Processed {n_windows} windows for trial {trial+1}")
            
            print(f"\nTotal training samples: {len(X_train)}")
            print(f"Training labels distribution: {np.bincount(y_train)}")
            
            print("\nTraining SVM...")
            # Train SVM
            X_train = np.array(X_train)
            y_train = np.array(y_train)
            print(f"Training data shape: {X_train.shape}")
            print(f"Training labels: {np.unique(y_train, return_counts=True)}")
            
            # Debug: Check correlation patterns in training data
            print("\nTraining data correlation patterns:")
            for label in [0, 1]:
                label_data = X_train[y_train == label]
                print(f"\nLabel {label}:")
                print(f"Mean correlations: {label_data.mean(axis=0)}")
                print(f"Std correlations: {label_data.std(axis=0)}")
                print(f"Number of samples: {len(label_data)}")
            
            # Calculate class weights
            class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
            class_weight_dict = dict(zip(np.unique(y_train), class_weights))
            print(f"Class weights: {class_weight_dict}")
            
            # Train SVM with class weights
            svm = SVC(kernel='linear', probability=True, class_weight=class_weight_dict)
            svm.fit(X_train, y_train)
            
            # Print SVM details
            print(f"SVM coefficients shape: {svm.coef_.shape}")
            print(f"SVM coefficients mean: {np.mean(np.abs(svm.coef_)):.6f}")
            print(f"SVM intercept: {svm.intercept_[0]:.6f}")
            
            # Evaluate on training set
            train_pred = svm.predict(X_train)
            train_prob = svm.predict_proba(X_train)
            print("\nTraining set metrics:")
            print(f"Accuracy: {accuracy_score(y_train, train_pred):.4f}")
            print(f"Precision: {precision_score(y_train, train_pred):.4f}")
            print(f"Recall: {recall_score(y_train, train_pred):.4f}")
            print(f"F1-score: {f1_score(y_train, train_pred):.4f}")
            print("\nConfusion Matrix:")
            print(confusion_matrix(y_train, train_pred))
            
            print("\nProcessing test subject", test_subj + 1)
            # Test
            n_correct = 0
            n_total = 0
            for trial in tqdm(range(n_trials), desc="Test trials", position=2, leave=False):
                if np.isnan(eeg_tensor[test_subj, trial]).all():
                    continue
                
                trial_correct = 0
                trial_total = 0
                
                eeg_feat = eeg_tensor[test_subj, trial]
                env0, env1 = load_audio_features(test_subj, trial, n_trials)
                min_len = min(eeg_feat.shape[0], env0.shape[0], env1.shape[0])
                eeg_feat = eeg_feat[:min_len]
                env0 = env0[:min_len]
                env1 = env1[:min_len]
                
                # Collect predictions and true labels for the trial
                trial_true_labels = []
                trial_preds = []
                
                # Process windows
                n_windows = 0
                for start in range(0, min_len - window_len + 1, stride):
                    end = start + window_len
                    eeg_win = eeg_feat[start:end]
                    env0_win = env0[start:end]
                    env1_win = env1[start:end]
                    # Compute CCA correlations for both streams (with PCA)
                    corrs = [
                        compute_window_correlations(eeg_win, env0_win),
                        compute_window_correlations(eeg_win, env1_win)
                    ]
                    # Get SVM prediction
                    corrs = np.array(corrs).reshape(1, -1)  # Reshape to 2D array
                    probs = svm.predict_proba(corrs)[0]
                    pred = svm.predict(corrs)[0]
                    true_label = labels[test_subj, trial] - 1
                    
                    # Debug prints for first few windows
                    if n_windows < 3:
                        print(f"\nWindow {n_windows + 1} debug:")
                        print(f"Correlations: {corrs[0]}")
                        print(f"Probabilities: {probs}")
                        print(f"Prediction: {pred}, True: {true_label}")
                    
                    # Add results for both streams
                    for stream_idx in range(2):
                        results.append({
                            'experiment': exp_name,
                            'subject': test_subj+1,
                            'trial': trial+1,
                            'window_start': start,
                            'window_end': end,
                            'stream': stream_idx+1,
                            'true_label': true_label+1,
                            'predicted_label': pred+1,
                            'prob_0': probs[0],
                            'prob_1': probs[1],
                        })
                    if pred == true_label:
                        n_correct += 1
                        trial_correct += 1
                    n_total += 1
                    trial_total += 1
                    n_windows += 1
                    
                    trial_true_labels.append(true_label)
                    trial_preds.append(pred)
                
                # Calculate and print only accuracy for the trial
                accuracy = accuracy_score(trial_true_labels, trial_preds)
                print(f"\nTrial {trial+1} accuracy: {accuracy:.4f} ({trial_correct}/{trial_total})")
            
            print(f"\nOverall accuracy: {n_correct/n_total:.4f} ({n_correct}/{n_total})")
    
    # Save results
    print(f"\nNumber of results collected: {len(results)}")
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(OUTPUT_DIR, 'cca_attention_decoding_results_full.csv'), index=False)

if __name__ == '__main__':
    run_decoding() 