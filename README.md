# Attention Decoding Pipeline

This repository contains a comprehensive pipeline for decoding auditory attention from EEG data using machine learning techniques. The pipeline includes data preprocessing, feature extraction, model training, and evaluation components.

## Dataset

This project uses the Auditory Attention Detection Dataset from KU Leuven, which contains EEG recordings from 16 normal-hearing subjects. The dataset was collected in a soundproof, electromagnetically shielded room using a 64-channel BioSemi ActiveTwo system at a sampling rate of 8196 Hz.

Dataset Reference:
- Biesmans, W., Das, N., Francart, T., & Bertrand, A. (2016). Auditory-inspired speech envelope extraction methods for improved EEG-based auditory attention detection in a cocktail party scenario. IEEE Transactions on Neural Systems and Rehabilitation Engineering, 25(5), 402-412.
- Dataset DOI: [10.5281/zenodo.4004271](https://zenodo.org/records/4004271)

## Features

- EEG data preprocessing and cleaning
- Feature extraction and selection
- Machine learning model training and evaluation
- Cross-validation with proper trial separation
- Performance visualization and analysis
- Support for parallel processing

## Requirements

The project requires Python 3.7+ and the following packages:
```
numpy>=1.21.0
pandas>=1.3.0
scipy>=1.7.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
mne>=1.0.0
joblib>=1.1.0
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Attention_Decoding_Pipeline.git
cd Attention_Decoding_Pipeline
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Download the dataset from [Zenodo](https://zenodo.org/records/4004271)
2. Place the dataset files in the appropriate directory
3. Run the preprocessing script:
```bash
python preprocess_data.py
```
4. Train and evaluate the model:
```bash
python train_model.py
```

## Project Structure

```
Attention_Decoding_Pipeline/
├── data/                   # Data directory
├── models/                 # Saved models
├── results/               # Results and visualizations
├── src/                   # Source code
│   ├── preprocessing/     # Data preprocessing modules
│   ├── features/         # Feature extraction modules
│   ├── models/           # Model training modules
│   └── utils/            # Utility functions
├── requirements.txt       # Project dependencies
└── README.md             # Project documentation
```

## Important Notes

- The pipeline implements proper cross-validation to avoid trial-specific overfitting
- Results are saved in the `results` directory
- Model checkpoints are saved in the `models` directory

## License

This project is licensed under the MIT License - see the LICENSE file for details.
``` 