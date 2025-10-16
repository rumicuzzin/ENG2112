"""
Configuration file for ENG2112 Machine Learning Project
Centralized settings for data paths, model parameters, and experiment configuration
"""
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = OUTPUT_DIR / "models"
RESULTS_DIR = OUTPUT_DIR / "results"
FIGURES_DIR = OUTPUT_DIR / "figures"

# Ensure output directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Data configuration
DATA_FILE = DATA_DIR / "ai4i2020.csv"
TARGET_COLUMN = "Machine failure"
SENSOR_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

# Train/test split configuration
TEST_SIZE = 0.2
RANDOM_STATE = 42
STRATIFY = True

# Cross-validation configuration
CV_FOLDS = 5
CV_SHUFFLE = True

# Model hyperparameters
MODEL_PARAMS = {
    "BalancedRF": {
        "n_estimators": 600,
        "max_depth": None,
        "random_state": RANDOM_STATE,
        "n_jobs": -1
    },
    "EasyEnsemble": {
        "n_estimators": 10,
        "random_state": RANDOM_STATE,
        "n_jobs": -1
    },
    "SMOTE_LogReg": {
        "smote_random_state": RANDOM_STATE,
        "max_iter": 2000,
        "class_weight": None
    },
    "ADASYN_LogReg": {
        "adasyn_random_state": RANDOM_STATE,
        "max_iter": 2000,
        "class_weight": None
    },
    "ClassWeight_LogReg": {
        "max_iter": 2000,
        "class_weight": "balanced"
    },
    "ClassWeight_RF": {
        "n_estimators": 600,
        "class_weight": "balanced_subsample",
        "random_state": RANDOM_STATE,
        "n_jobs": -1
    }
}

# Evaluation configuration
THRESHOLD_PREFERENCE = "f1"
TOP_N_MODELS = 3

# Output file names
RESULTS_CSV = RESULTS_DIR / "failure_imbalance_comparison.csv"
PR_CURVE_PLOT = FIGURES_DIR / "pr_curves_top3.png"
ROC_CURVE_PLOT = FIGURES_DIR / "roc_curves_top3.png"

# Logging configuration
VERBOSE = True