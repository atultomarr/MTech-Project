import pickle
import numpy as np
import os

def check_cca_outputs(subj=1):
    cca_dir = 'python_attention_pipeline/data/paper_replication/cca_outputs/'
    with open(os.path.join(cca_dir, f'subj{subj}_X_test_cca.pkl'), 'rb') as f:
        X_test_cca = pickle.load(f)
    with open(os.path.join(cca_dir, f'subj{subj}_Y_test_cca.pkl'), 'rb') as f:
        Y_test_cca = pickle.load(f)
    print(f'Checking CCA outputs for Subject {subj}')
    for i, (x, y) in enumerate(zip(X_test_cca, Y_test_cca)):
        print(f'  Test trial {i+1}:')
        for name, arr in [('EEG', x), ('Audio', y)]:
            n_nan = np.isnan(arr).sum()
            arr_min = np.nanmin(arr)
            arr_max = np.nanmax(arr)
            arr_mean = np.nanmean(arr)
            is_const = np.allclose(arr, arr[0])
            print(f'    {name}: shape={arr.shape}, NaNs={n_nan}, min={arr_min:.4f}, max={arr_max:.4f}, mean={arr_mean:.4f}, constant={is_const}')

if __name__ == '__main__':
    check_cca_outputs(1) 