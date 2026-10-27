"""
Central configuration for Host-Based Intrusion Detection System (HIDS).
All phases import configuration settings from this single file.
"""
import os
from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT  # contains Training_Data_Master, Validation_Data_Master, Attack_Data_Master
RESULTS_DIR = PROJECT_ROOT / "results"

# Standard counts for ADFA-LD dataset verification
EXPECTED_COUNTS = {
    "train": 833,
    "val": 4372,
    "attack": 746
}

# Windowing parameters
DEFAULT_L = 20
DEFAULT_S = 10

# Feature extraction parameters
DEFAULT_B_BI = 256
DEFAULT_B_TRI = 256

# Model & Evaluation parameters
DEFAULT_SEEDS = [0, 1, 2, 3, 4]
DEMO_SEEDS = [0]
PERCENTILES = [99.0, 95.0]
SVM_MAX_SAMPLES = 5000
AE_EPOCHS = 50
DEMO_AE_EPOCHS = 10

# Phase definitions with weights and expected key output files
PHASE_REGISTRY = {
    1: {
        "title": "Data loading and verification",
        "script": "phase1_data_loading.py",
        "weight": 8,
        "key_file": "results/phase1/dataset_stats.csv",
        "summary": "results/phase1/summary.md"
    },
    2: {
        "title": "Preprocessing and features",
        "script": "phase2_preprocessing_features.py",
        "weight": 12,
        "key_file": "results/phase2/features_summary.csv",
        "summary": "results/phase2/summary.md"
    },
    3: {
        "title": "Baseline detectors: Isolation Forest and One-Class SVM",
        "script": "phase3_baselines.py",
        "weight": 12,
        "key_file": "results/phase3/baseline_metrics.csv",
        "summary": "results/phase3/summary.md"
    },
    4: {
        "title": "Proposed lightweight autoencoder",
        "script": "phase4_autoencoder.py",
        "weight": 25,
        "key_file": "results/phase4/autoencoder_metrics.csv",
        "summary": "results/phase4/summary.md"
    },
    5: {
        "title": "LSTM next-call baseline in PyTorch",
        "script": "phase5_lstm.py",
        "weight": 13,
        "key_file": "results/phase5/lstm_metrics.csv",
        "summary": "results/phase5/summary.md"
    },
    6: {
        "title": "Evaluation and resource measurement",
        "script": "phase6_evaluation_resources.py",
        "weight": 18,
        "key_file": "results/phase6/comparison_table.csv",
        "summary": "results/phase6/summary.md"
    },
    7: {
        "title": "LID-DS 2021 evaluation",
        "script": "phase7_lidds.py",
        "weight": 7,
        "key_file": "results/phase7/lidds_metrics.csv",
        "summary": "results/phase7/summary.md"
    },
    8: {
        "title": "Ablation and final results package",
        "script": "phase8_final_package.py",
        "weight": 5,
        "key_file": "results/final/README.md",
        "summary": "results/phase8/summary.md"
    }
}
