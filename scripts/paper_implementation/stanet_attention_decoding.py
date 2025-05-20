print("Script starting...")

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
from tqdm.auto import tqdm
import soundfile as sf
from scipy.signal import hilbert
import warnings
warnings.filterwarnings('ignore')

print("Imports completed")

# Debug CUDA setup
print("\n=== CUDA Setup ===")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Current CUDA device: {torch.cuda.current_device()}")
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")

# Parameters
EEG_PATH = '/data/home1/rishavs/atul/cocoha-matlab-toolbox-master/cocoha-matlab-toolbox-master/python_attention_pipeline/data/eeg_tensor_filterbank.npy'
AUDIO_DIR = '/data/home1/rishavs/atul/cocoha-matlab-toolbox-master/cocoha-matlab-toolbox-master/python_attention_pipeline/data/aligned_audio'
LABELS_PATH = '/data/home1/rishavs/atul/cocoha-matlab-toolbox-master/cocoha-matlab-toolbox-master/python_attention_pipeline/data/labels.npy'
OUTPUT_DIR = '/data/home1/rishavs/atul/cocoha-matlab-toolbox-master/cocoha-matlab-toolbox-master/python_attention_pipeline/data/paper_replication/decoding_results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Experiment settings
EXPERIMENTS = {
    'exp1': {'window_sec': 8.0, 'stride_sec': 0.25},
    'exp2': {'window_sec': 6.5, 'stride_sec': 0.25},
}
DOWNSAMPLED_FS = 128
N_PCA = 0.99  # keep 99% variance
N_CCA = 1     # number of CCA components

# Set number of GPUs to use
NUM_GPUS = torch.cuda.device_count()
print(f"\nNumber of available GPUs: {NUM_GPUS}")
for i in range(NUM_GPUS):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

# Set device
if torch.cuda.is_available():
    if NUM_GPUS > 1:
        # Use all available GPUs
        device = torch.device('cuda')
        print(f"\nUsing all {NUM_GPUS} GPUs in parallel")
    else:
        device = torch.device('cuda:0')
        print("\nUsing single GPU")
else:
    device = torch.device('cpu')
    print("\nUsing CPU")

def debug_tensor_info(tensor, name=""):
    """Debug function to print tensor information"""
    print(f"\n{name} Tensor Info:")
    print(f"Shape: {tensor.shape}")
    print(f"Device: {tensor.device}")
    print(f"Type: {tensor.dtype}")
    print(f"Requires grad: {tensor.requires_grad}")
    if tensor.device.type == 'cuda':
        print(f"GPU Memory: {tensor.element_size() * tensor.nelement() / 1024**2:.2f} MB")

def debug_gpu_memory():
    """Print GPU memory usage for all available GPUs"""
    print("\n=== GPU Memory Usage ===")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"Allocated: {torch.cuda.memory_allocated(i) / 1024**2:.2f} MB")
        print(f"Cached: {torch.cuda.memory_reserved(i) / 1024**2:.2f} MB")

def load_audio_features(subject_idx, trial_idx, n_trials):
    """Load audio envelope features for a subject/trial."""
    env0_path = os.path.join(AUDIO_DIR, f'S{subject_idx+1}_trial{trial_idx+1}_mixture_0_aligned.wav')
    env1_path = os.path.join(AUDIO_DIR, f'S{subject_idx+1}_trial{trial_idx+1}_mixture_1_aligned.wav')
    env0, fs = sf.read(env0_path)
    env1, fs = sf.read(env1_path)
    env0 = np.abs(hilbert(env0))
    env1 = np.abs(hilbert(env1))
    return env0, env1

def compute_window_correlations(eeg_win, env_win):
    """Compute CCA correlations for a window of data."""
    # PCA for EEG and audio
    pca_eeg = PCA(n_components=N_PCA)
    pca_audio = PCA(n_components=N_PCA)
    eeg_win_pca = pca_eeg.fit_transform(eeg_win)
    env_win_pca = pca_audio.fit_transform(env_win.reshape(-1, 1))
    
    # CCA
    cca = CCA(n_components=N_CCA)
    eeg_win_cca, env_win_cca = cca.fit_transform(eeg_win_pca, env_win_pca)
    return np.corrcoef(eeg_win_cca[:, 0], env_win_cca[:, 0])[0, 1]

def process_trial(eeg_feat, env0, env1, window_len, stride, label):
    """Process a single trial and extract features."""
    min_len = min(eeg_feat.shape[0], env0.shape[0], env1.shape[0])
    eeg_feat = eeg_feat[:min_len]
    env0 = env0[:min_len]
    env1 = env1[:min_len]
    
    features = []
    labels = []
    
    for start in range(0, min_len - window_len + 1, stride):
        end = start + window_len
        eeg_win = eeg_feat[start:end]
        env0_win = env0[start:end]
        env1_win = env1[start:end]
        
        # Compute CCA correlations for both streams
        corrs = [
            compute_window_correlations(eeg_win, env0_win),
            compute_window_correlations(eeg_win, env1_win)
        ]
        features.append(corrs)
        labels.append(label)
    
    return features, labels

# Custom Dataset class
class AttentionDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        print("\nDataset Info:")
        print(f"Features shape: {self.features.shape}")
        print(f"Labels shape: {self.labels.shape}")
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# Modified Stanet model for binary classification
class AttentionStanet(nn.Module):
    def __init__(self, input_size=2):
        super(AttentionStanet, self).__init__()
        
        # spatial attention
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(128, 1), stride=(1, 1))
        self.pooling1 = torch.nn.MaxPool2d(kernel_size=(1, 1), stride=(1, 1))
        self.linear1 = nn.Linear(input_size, 8)  # Use input_size instead of hardcoded 20
        self.dropout = 0.5

        self.elu = nn.Sequential(
            nn.BatchNorm2d(num_features=1),
            nn.ELU(),
            nn.Dropout(p=self.dropout)
        )

        self.linear2 = nn.Linear(8, input_size)  # Use input_size instead of hardcoded 20

        # conv block
        self.conv2 = nn.Conv2d(in_channels=input_size, out_channels=5, kernel_size=(1, 1), stride=(1, 1))
        self.pooling2 = torch.nn.MaxPool2d(kernel_size=(1, 1), stride=(4, 1))

        self.tanh = nn.Sequential(
            nn.BatchNorm2d(num_features=1),
            nn.Tanh(),
            nn.Dropout(p=self.dropout)
        )

        self.attention = MultiHeadAttention(key_size=5, query_size=5,
                                          value_size=5, num_hiddens=5, num_heads=1, dropout=self.dropout)

        self.fc1 = nn.Linear(160, 2)  # Changed to 2 for binary classification
        
    def forward(self, x):
        # Debug prints for forward pass
        print("\n=== Forward Pass Debug ===")
        print(f"Input device: {x.device}")
        print(f"Input shape: {x.shape}")
        print(f"Model device: {next(self.parameters()).device}")
        
        # Reshape input to match Stanet's expected format
        x = x.unsqueeze(1)  # Add channel dimension
        x = x.unsqueeze(2)  # Add height dimension
        print(f"Reshaped input shape: {x.shape}")
        
        # Check if input is on GPU
        if x.device.type == 'cuda':
            print("Input is on GPU")
            print(f"GPU memory before forward: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        else:
            print("WARNING: Input is not on GPU!")
        
        # Run forward pass
        R_c = self.conv1(x)
        R_s = self.pooling1(self.elu(R_c))
        M_s = self.linear2(self.elu(self.linear1(R_s)))
        
        Ep = M_s * x
        
        Ep = Ep.permute(0, 3, 2, 1)
        Epc = self.conv2(Ep)
        Epc = Epc.permute(0, 3, 2, 1)
        Eps = self.pooling2(self.tanh(Epc))
        
        Eps = Eps.squeeze(dim=1)
        E_t = self.attention(Eps, Eps, Eps)
        
        E_t = E_t.reshape(E_t.shape[0], -1)
        output = self.fc1(E_t)
        
        # Check output
        print(f"Output device: {output.device}")
        print(f"Output shape: {output.shape}")
        if output.device.type == 'cuda':
            print("Output is on GPU")
            print(f"GPU memory after forward: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        else:
            print("WARNING: Output is not on GPU!")
        
        return output

def train_model(model, train_loader, criterion, optimizer, device, num_epochs=50):
    print("\n=== Model Training Info ===")
    print(f"Model device: {next(model.parameters()).device}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Batch size: {train_loader.batch_size}")
    
    # Print initial GPU memory state
    debug_gpu_memory()
    
    best_val_acc = 0
    best_model = None
    
    # Create progress bar for epochs
    epoch_pbar = tqdm(range(num_epochs), desc='Training Progress', position=0)
    
    for epoch in epoch_pbar:
        # Training phase
        model.train()
        train_loss = 0
        train_preds = []
        train_labels = []
        
        # Create progress bar for training batches
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]', 
                         position=1, leave=False)
        
        for batch_idx, (features, labels) in enumerate(train_pbar):
            # Debug first batch
            if batch_idx == 0:
                print("\nFirst batch info:")
                debug_tensor_info(features, "Features")
                debug_tensor_info(labels, "Labels")
                debug_gpu_memory()
            
            # Move data to GPU and verify
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            if batch_idx == 0:
                print("\nAfter moving to GPU:")
                debug_tensor_info(features, "Features")
                debug_tensor_info(labels, "Labels")
                debug_gpu_memory()
            
            optimizer.zero_grad()
            
            # Check if model is on GPU before forward pass
            if batch_idx == 0:
                print("\nModel parameters device check:")
                for name, param in model.named_parameters():
                    print(f"{name}: {param.device}")
            
            outputs = model(features)
            
            # Debug first batch outputs
            if batch_idx == 0:
                debug_tensor_info(outputs, "Model Outputs")
                debug_gpu_memory()
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            if batch_idx == 0:
                print("\nAfter backward pass:")
                debug_gpu_memory()
            
            train_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
            
            # Update training progress bar
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            # Only process first batch for debugging
            if batch_idx == 0:
                break
        
        train_acc = accuracy_score(train_labels, train_preds)
        
        # Update epoch progress bar
        epoch_pbar.set_postfix({
            'train_loss': f'{train_loss/len(train_loader):.4f}',
            'train_acc': f'{train_acc:.4f}'
        })
        
        # Save best model
        if train_acc > best_val_acc:
            best_val_acc = train_acc
            best_model = model.state_dict()
    
    return best_model

def main():
    print("\n=== Starting Main ===")
    print("Loading data...")
    
    # Print initial GPU memory state
    debug_gpu_memory()
    
    print("Loading EEG tensor...")
    eeg_tensor = np.load(EEG_PATH, mmap_mode='r')
    print("Loading labels...")
    labels = np.load(LABELS_PATH)
    n_subjects, n_trials, n_samples, n_channels = eeg_tensor.shape
    print(f"EEG tensor shape: {eeg_tensor.shape}")
    print(f"Labels shape: {labels.shape}")
    
    # Process each experiment
    exp_pbar = tqdm(EXPERIMENTS.items(), desc='Experiments', position=0)
    for exp_name, exp_params in exp_pbar:
        print(f"\nProcessing experiment: {exp_name}")
        exp_pbar.set_description(f'Experiment: {exp_name}')
        window_len = int(exp_params['window_sec'] * DOWNSAMPLED_FS)
        stride = int(exp_params['stride_sec'] * DOWNSAMPLED_FS)
        print(f"Window length: {window_len}, Stride: {stride}")
        
        # Process each subject
        subj_pbar = tqdm(range(n_subjects), desc='Subjects', position=1)
        for test_subj in subj_pbar:
            print(f"\nProcessing subject {test_subj + 1}")
            subj_pbar.set_description(f'Subject {test_subj + 1}')
            
            # Prepare training data
            print("Preparing training data...")
            X_train, y_train = [], []
            
            # Process training data
            print("Processing training trials...")
            for subj in range(n_subjects):
                if subj == test_subj:
                    continue
                    
                print(f"Processing subject {subj + 1} trials...")
                for trial in range(n_trials):
                    if np.isnan(eeg_tensor[subj, trial]).all():
                        continue
                    print(f"Processing trial {trial + 1}")
                    eeg_feat = eeg_tensor[subj, trial]
                    env0, env1 = load_audio_features(subj, trial, n_trials)
                    trial_features, trial_labels = process_trial(
                        eeg_feat, env0, env1, window_len, stride, labels[subj, trial] - 1
                    )
                    X_train.extend(trial_features)
                    y_train.extend(trial_labels)
            
            print("Converting to numpy arrays...")
            X_train = np.array(X_train)
            y_train = np.array(y_train)
            
            print("Normalizing data...")
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            
            print("Creating dataset...")
            train_dataset = AttentionDataset(X_train, y_train)
            
            print("Initializing model...")
            model = AttentionStanet(input_size=X_train.shape[1])
            print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
            
            # Move model to GPU
            print("Moving model to GPU...")
            if NUM_GPUS > 1:
                model = model.to(device)
                model = nn.DataParallel(model, device_ids=list(range(NUM_GPUS)))
            else:
                model = model.to(device)
            
            print("Creating data loader...")
            batch_size = 32 * NUM_GPUS
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                                    num_workers=4, pin_memory=True)
            
            print("Setting up loss and optimizer...")
            criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([
                len(y_train) / (2 * np.sum(y_train == 0)),
                len(y_train) / (2 * np.sum(y_train == 1))
            ]).to(device))
            
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            
            print("Starting training...")
            best_model = train_model(model, train_loader, criterion, optimizer, device)
            
            print("Training complete!")
            break  # For debugging, only process first subject
        break  # For debugging, only process first experiment

if __name__ == "__main__":
    main() 