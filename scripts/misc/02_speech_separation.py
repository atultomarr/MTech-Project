import os
import torch
import torchaudio
from speechbrain.inference.separation import SepformerSeparation as separator
import argparse
import torch.multiprocessing as mp
from multiprocessing import Queue
from queue import Empty
from tqdm import tqdm
import gc
import time
import math

# Input and output directories
MIX_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'mixtures')
SEP_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'separated')

# Create directories if they don't exist
os.makedirs(MIX_DIR, exist_ok=True)
os.makedirs(SEP_DIR, exist_ok=True)

print(f"Mixtures directory: {MIX_DIR}")
print(f"Separated directory: {SEP_DIR}")

def clear_gpu_memory():
    """Clear GPU memory."""
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def resample_audio(input_path, target_sr=8000):
    """Resample audio to target sample rate."""
    waveform, sr = torchaudio.load(input_path)
    
    # Convert to mono if needed
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    
    # Resample if needed
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)
    
    return waveform, target_sr

def process_file(file_path, gpu_id):
    """Process a single file on a specific GPU."""
    try:
        torch.cuda.set_device(gpu_id)
        print(f"Processing {file_path} on GPU {gpu_id}")

        # Load and resample audio
        waveform, sr = resample_audio(file_path)
        chunk_size_sec = 10  # seconds
        chunk_size = chunk_size_sec * sr
        total_samples = waveform.shape[1]
        num_chunks = math.ceil(total_samples / chunk_size)
        base_name = os.path.splitext(os.path.basename(file_path))[0]

        # Load model on specific GPU
        model = separator.from_hparams(
            source="speechbrain/sepformer-wsj02mix",
            savedir='pretrained_models/sepformer-wsj02mix',
            run_opts={"device": f"cuda:{gpu_id}"}
        )

        separated_chunks = []
        for i in range(num_chunks):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, total_samples)
            chunk = waveform[:, start:end]
            temp_path = f"temp_resampled_{gpu_id}.wav"
            torchaudio.save(temp_path, chunk, sr)
            est_sources = model.separate_file(path=temp_path)
            separated_chunks.append(est_sources)
            if os.path.exists(temp_path):
                os.remove(temp_path)
            torch.cuda.empty_cache()

        # Concatenate all chunks
        source_0 = torch.cat([chunk[:, :, 0] for chunk in separated_chunks], dim=1)
        source_1 = torch.cat([chunk[:, :, 1] for chunk in separated_chunks], dim=1)

        torchaudio.save(
            os.path.join(SEP_DIR, f"{base_name}_0.wav"),
            source_0.detach().cpu(),
            8000
        )
        torchaudio.save(
            os.path.join(SEP_DIR, f"{base_name}_1.wav"),
            source_1.detach().cpu(),
            8000
        )

        del model
        del separated_chunks
        clear_gpu_memory()
        print(f"Successfully separated {file_path} on GPU {gpu_id}")

    except Exception as e:
        print(f"Error processing {file_path} on GPU {gpu_id}: {str(e)}")
        import traceback
        traceback.print_exc()
        clear_gpu_memory()

def gpu_worker(gpu_id, file_queue):
    """Worker function for each GPU."""
    while True:
        try:
            file_path = file_queue.get_nowait()
            process_file(file_path, gpu_id)
        except Empty:
            break
        except Exception as e:
            print(f"Error in GPU {gpu_id} worker: {str(e)}")
            continue

def process_files_parallel(files, num_gpus):
    """Process files in parallel across multiple GPUs."""
    # Create a queue of files to process
    file_queue = Queue()
    for file in files:
        file_queue.put(os.path.join(MIX_DIR, file))
    
    # Create and start processes for each GPU
    processes = []
    for gpu_id in range(num_gpus):
        p = mp.Process(target=gpu_worker, args=(gpu_id, file_queue))
        p.start()
        processes.append(p)
    
    # Wait for all processes to complete
    for p in processes:
        p.join()

def main():
    parser = argparse.ArgumentParser(description='Process audio files for speech separation')
    parser.add_argument('--test', action='store_true', help='Process only one file for testing')
    args = parser.parse_args()

    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)
    
    # Get number of available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Found {num_gpus} GPUs")
    
    # Print GPU information
    for i in range(num_gpus):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
    
    # List available files
    if os.path.exists(MIX_DIR):
        files = [f for f in os.listdir(MIX_DIR) if f.endswith('.wav')]
        if files:
            files.sort()
            
            if args.test:
                files = files[:1]
                print("\nRunning in test mode - processing only one file")
            
            print(f"\nProcessing {len(files)} files using {num_gpus} GPUs")
            process_files_parallel(files, num_gpus)
            
            print("\nProcessing completed!")
        else:
            print("No .wav files found in the mixtures directory!")
    else:
        print(f"Mixtures directory not found: {MIX_DIR}")

if __name__ == "__main__":
    main() 