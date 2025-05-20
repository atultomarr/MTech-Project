import os
import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import hilbert
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import logging
from collections import Counter
from tqdm import tqdm

# Set random seed for reproducibility
np.random.seed(42)

# Configure logging
logging.basicConfig(
    filename=os.path.join('/data/home1/rishavs/atul/cocoha-matlab-toolbox-master/cocoha-matlab-toolbox-master/python_attention_pipeline/data/paper_replication/decoding_results', 'decoding.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# File paths
EEG_PATH = '/data/home1/rishavs/atul/cocoha-matlab-toolbox-master/cocoha-matlab-toolbox-master/python_attention_pipeline/data/eeg_tensor_filterbank.npy'
AUDIO_DIR = '/data/home1/rishavs/atul/cocoha-matlab-toolbox-master/cocoha-matlab-toolbox-master/python_attention_pipeline/data/aligned_audio'
LABELS_PATH = '/data/home1/rishavs/atul/cocoha-matlab-toolbox-master/cocoha-matlab-toolbox-master/python_attention_pipeline/data/labels.npy'
OUTPUT_DIR = '/data/home1/rishavs/atul/cocoha-matlab-toolbox-master/cocoha-matlab-toolbox-master/python_attention_pipeline/data/paper_replication/decoding_results'

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Constants
FS = 128  # Downsampled frequency

# Define multiple experiments with optimized window sizes and strides
EXPERIMENTS = {
    'exp1': {'window_sec': 8.0, 'stride_sec': 1},    # Original exp
}

# Classifier configurations
CLASSIFIERS = {
    'svm': {
        'model': SVC(probability=True, random_state=42),
        'param_grid': {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf']
        }
    },
    'xgboost': {
        'model': XGBClassifier(random_state=42),
        'param_grid': {
            'max_depth': [3, 5],
            'learning_rate': [0.01, 0.1],
            'n_estimators': [100, 200]
        }
    }
}

def compute_window_correlations(eeg_win, env0_win, env1_win, n_cca=1, n_pca=0.99):
    """Compute CCA correlations between EEG and audio envelopes with NaN handling."""
    try:
        # Check for NaNs
        if np.any(np.isnan(eeg_win)) or np.any(np.isnan(env0_win)) or np.any(np.isnan(env1_win)):
            logging.warning("NaN detected in window inputs")
            return np.array([[np.nan, np.nan]])

        # Standardize inputs
        scaler = StandardScaler()
        eeg_win = scaler.fit_transform(eeg_win)
        env0_win = scaler.fit_transform(env0_win.reshape(-1, 1)).flatten()
        env1_win = scaler.fit_transform(env1_win.reshape(-1, 1)).flatten()

        # PCA on EEG
        pca_eeg = PCA(n_components=n_pca)
        eeg_pca = pca_eeg.fit_transform(eeg_win)
        # PCA on envelopes (single component)
        pca_env = PCA(n_components=1)
        env0_pca = pca_env.fit_transform(env0_win.reshape(-1, 1)).flatten()
        env1_pca = pca_env.fit_transform(env1_win.reshape(-1, 1)).flatten()

        # CCA
        cca = CCA(n_components=n_cca, max_iter=1000)
        cca.fit(eeg_pca, env0_pca.reshape(-1, 1))
        eeg_c0, env0_c = cca.transform(eeg_pca, env0_pca.reshape(-1, 1))
        cca.fit(eeg_pca, env1_pca.reshape(-1, 1))
        eeg_c1, env1_c = cca.transform(eeg_pca, env1_pca.reshape(-1, 1))

        # Compute correlations
        corr_env0 = np.corrcoef(eeg_c0.T, env0_c.T)[0, 1]
        corr_env1 = np.corrcoef(eeg_c1.T, env1_c.T)[0, 1]
        if np.isnan(corr_env0) or np.isnan(corr_env1):
            logging.warning("NaN in CCA correlations")
            return np.array([[np.nan, np.nan]])
        return np.array([[corr_env0, corr_env1]])
    except Exception as e:
        logging.error(f"CCA failed: {str(e)}")
        return np.array([[np.nan, np.nan]])

def load_and_preprocess_data():
    try:
        # Load EEG data
        eeg_tensor = np.load(EEG_PATH, mmap_mode='r')
        labels = np.load(LABELS_PATH)
        logging.info(f"EEG tensor shape: {eeg_tensor.shape}")
        logging.info(f"Labels shape: {labels.shape}")

        # Adjust labels to 0/1 if necessary
        if labels.min() != 0:
            labels = labels - 1
            logging.info("Labels adjusted to 0/1")

        # Load audio
        audio_files = [f for f in os.listdir(AUDIO_DIR) if f.endswith('_aligned.wav')]
        if len(audio_files) < 2:
            logging.error("Insufficient audio files found")
            raise ValueError("Expected at least two audio files")
        
        # Group audio files by subject and trial
        audio_data = {}
        for f in audio_files:
            # Parse filename: S{subject}_trial{trial}_mixture_{mixture}_aligned.wav
            parts = f.split('_')
            if len(parts) != 5:
                logging.warning(f"Skipping file with unexpected format: {f}")
                continue
                
            subject = int(parts[0][1:])  # Remove 'S' prefix
            trial = int(parts[1][5:])    # Remove 'trial' prefix
            mixture = int(parts[3])      # mixture number
            
            key = f'S{subject}_trial{trial}'
            if key not in audio_data:
                audio_data[key] = {}
            
            sr, wav = wavfile.read(os.path.join(AUDIO_DIR, f))
            if sr != FS:
                logging.warning(f"Audio {f} sample rate {sr} != {FS}")
            if np.any(np.isnan(wav)):
                logging.warning(f"Audio {f} contains NaN values")
            audio_data[key][mixture] = wav
            
        logging.info(f"Audio files loaded for {len(audio_data)} subject-trials")
        return eeg_tensor, labels, audio_data
    except Exception as e:
        logging.error(f"Data loading failed: {str(e)}")
        raise

def train_classifier(X, y, classifier_name, validation_split=0.2):
    """Train classifier with grid search and handle imbalance."""
    try:
        # Check for NaNs in features
        if np.any(np.isnan(X)) or np.any(np.isnan(y)):
            logging.error("NaN detected in training data")
            raise ValueError("Training data contains NaN")

        # Log class distribution
        class_counts = Counter(y)
        logging.info(f"Class distribution: {class_counts}")

        # Apply SMOTE for imbalance
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        logging.info(f"After SMOTE: {Counter(y_resampled)}")

        # Split for validation
        n_val = int(len(X_resampled) * validation_split)
        X_train, X_val = X_resampled[:-n_val], X_resampled[-n_val:]
        y_train, y_val = y_resampled[:-n_val], y_resampled[-n_val:]

        # Grid search
        clf_config = CLASSIFIERS[classifier_name]
        grid = GridSearchCV(
            clf_config['model'],
            clf_config['param_grid'],
            cv=5,
            scoring='accuracy',
            n_jobs=-1
        )
        grid.fit(X_train, y_train)
        logging.info(f"Best {classifier_name} params: {grid.best_params_}")

        # Evaluate on validation set
        val_preds = grid.predict(X_val)
        val_acc = accuracy_score(y_val, val_preds)
        val_precision = precision_score(y_val, val_preds, average='binary')
        val_recall = recall_score(y_val, val_preds, average='binary')
        val_f1 = f1_score(y_val, val_preds, average='binary')
        logging.info(f"Validation metrics: Accuracy={val_acc:.4f}, Precision={val_precision:.4f}, Recall={val_recall:.4f}, F1={val_f1:.4f}")

        return grid.best_estimator_
    except Exception as e:
        logging.error(f"Classifier training failed: {str(e)}")
        return None

def main():
    # Load data
    eeg_tensor, labels, audio_data = load_and_preprocess_data()
    n_subjects, n_trials, n_samples, n_channels = eeg_tensor.shape

    # Results storage
    all_results = []

    # Iterate over experiments
    for exp_name, exp_config in tqdm(EXPERIMENTS.items(), desc="Experiments"):
        window_sec = exp_config['window_sec']
        stride_sec = exp_config['stride_sec']
        window_samples = int(window_sec * FS)
        stride_samples = int(stride_sec * FS)
        logging.info(f"Running {exp_name}: window={window_sec}s, stride={stride_sec}s")

        for classifier_name in tqdm(CLASSIFIERS, desc="Classifiers", leave=False):
            logging.info(f"Using classifier: {classifier_name}")

            # Train/test split (subjects 0, 1 for training, 2 for testing)
            training_subjects = [0, 1]
            test_subjects = [2]

            # Collect training data
            X_train, y_train = [], []
            for subj in tqdm(training_subjects, desc="Training subjects", leave=False):
                for trial in tqdm(range(2), desc=f"Subject {subj+1} trials", leave=False):
                    eeg = eeg_tensor[subj, trial]
                    true_label = labels[subj, trial]
                    if np.isnan(eeg).all():
                        logging.warning(f"Skipping trial {trial+1}, subj {subj+1}: all NaN")
                        continue

                    # Get audio data for this subject and trial
                    key = f'S{subj+1}_trial{trial+1}'
                    if key not in audio_data or 0 not in audio_data[key] or 1 not in audio_data[key]:
                        logging.warning(f"Missing audio data for {key}")
                        continue

                    # Compute audio envelopes
                    env0 = np.abs(hilbert(audio_data[key][0]))
                    env1 = np.abs(hilbert(audio_data[key][1]))
                    if len(env0) < n_samples or len(env1) < n_samples:
                        logging.warning(f"Audio length mismatch for trial {trial+1}, subj {subj+1}")
                        continue

                    # Windowing
                    windows = range(0, n_samples - window_samples + 1, stride_samples)
                    for start in tqdm(windows, desc=f"Windows for trial {trial+1}", leave=False):
                        end = start + window_samples
                        eeg_win = eeg[start:end]
                        env0_win = env0[start:end]
                        env1_win = env1[start:end]

                        # NaN check
                        if np.any(np.isnan(eeg_win)) or np.any(np.isnan(env0_win)) or np.any(np.isnan(env1_win)):
                            logging.warning(f"Skipping window {start}-{end}, subj {subj+1}: contains NaN")
                            continue

                        # Compute correlations
                        corrs = compute_window_correlations(eeg_win, env0_win, env1_win)
                        if np.any(np.isnan(corrs)):
                            logging.warning(f"NaN correlations for window {start}-{end}, subj {subj+1}")
                            continue

                        X_train.append(corrs[0])
                        y_train.append(true_label)

            X_train = np.array(X_train)
            y_train = np.array(y_train)
            logging.info(f"Training data: {X_train.shape}, Labels: {Counter(y_train)}")

            # Train classifier
            clf = train_classifier(X_train, y_train, classifier_name)
            if clf is None:
                logging.error(f"Skipping {classifier_name} for {exp_name}: training failed")
                continue

            # Test on test subjects
            for test_subj in tqdm(test_subjects, desc="Test subjects", leave=False):
                n_correct, n_total = 0, 0
                for trial in tqdm(range(2), desc=f"Test subject {test_subj+1} trials", leave=False):
                    eeg = eeg_tensor[test_subj, trial]
                    true_label = labels[test_subj, trial]
                    if np.isnan(eeg).all():
                        logging.warning(f"Skipping test trial {trial+1}, subj {test_subj+1}: all NaN")
                        continue

                    # Get audio data for this subject and trial
                    key = f'S{test_subj+1}_trial{trial+1}'
                    if key not in audio_data or 0 not in audio_data[key] or 1 not in audio_data[key]:
                        logging.warning(f"Missing audio data for {key}")
                        continue

                    # Compute audio envelopes
                    env0 = np.abs(hilbert(audio_data[key][0]))
                    env1 = np.abs(hilbert(audio_data[key][1]))
                    if len(env0) < n_samples or len(env1) < n_samples:
                        logging.warning(f"Audio length mismatch for test trial {trial+1}, subj {test_subj+1}")
                        continue

                    preds, probs = [], []
                    windows = range(0, n_samples - window_samples + 1, stride_samples)
                    for start in tqdm(windows, desc=f"Test windows for trial {trial+1}", leave=False):
                        end = start + window_samples
                        eeg_win = eeg[start:end]
                        env0_win = env0[start:end]
                        env1_win = env1[start:end]

                        if np.any(np.isnan(eeg_win)) or np.any(np.isnan(env0_win)) or np.any(np.isnan(env1_win)):
                            logging.warning(f"Skipping test window {start}-{end}, subj {test_subj+1}: contains NaN")
                            continue

                        corrs = compute_window_correlations(eeg_win, env0_win, env1_win)
                        if np.any(np.isnan(corrs)):
                            logging.warning(f"NaN correlations for test window {start}-{end}, subj {test_subj+1}")
                            continue

                        pred = clf.predict(corrs)[0]
                        prob = clf.predict_proba(corrs)[0]
                        preds.append(pred)
                        probs.append(prob)

                        # Store results
                        all_results.append({
                            'experiment': exp_name,
                            'classifier': classifier_name,
                            'subject': test_subj + 1,
                            'trial': trial + 1,
                            'window_start': start,
                            'window_end': end,
                            'true_label': true_label + 1,
                            'predicted_label': pred + 1,
                            'prob_0': prob[0],
                            'prob_1': prob[1]
                        })

                    if preds:
                        # Majority vote for trial
                        trial_pred = np.bincount(preds).argmax()
                        n_correct += (trial_pred == true_label)
                        n_total += 1

                # Log test accuracy
                if n_total > 0:
                    accuracy = n_correct / n_total
                    logging.info(f"Test accuracy for subject {test_subj + 1}: {accuracy:.4f}")

    # Save results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join(OUTPUT_DIR, 'decoding_results.csv'), index=False)
    logging.info("Results saved to decoding_results.csv")

if __name__ == "__main__":
    main()