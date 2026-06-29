import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / 'datasets'
MODELS_DIR = BASE_DIR / 'models'
RESULTS_DIR = BASE_DIR / 'results'

MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Training Configuration
CONFIG = {
    'data_dir': str(DATA_DIR),
    'img_size': (224, 224),
    'batch_size': 16,
    'initial_epochs': 20,
    'fine_tune_epochs': 30,
    'initial_lr': 1e-3,
    'fine_tune_lr': 1e-5,
    'dropout_rate': 0.4,
    'l2_reg': 5e-5,
}
