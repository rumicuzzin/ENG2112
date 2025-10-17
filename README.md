# ENG2112 - Machine Learning for Predictive Maintenanc

My names Finn

A structured Python implementation of machine learning models for predictive maintenance using the AI4I 2020 dataset. This project predicts machine failures using multiple classification strategies to handle extreme class imbalance.

## 📋 Project Overview

**Dataset**: AI4I 2020 Predictive Maintenance Dataset  
**Task**: Binary classification of machine failures (~3.4% failure rate)  
**Features**: Sensor readings (temperature, speed, torque, tool wear)

## 🗂️ Project Structure

```
ENG2112/
├── src/
│   ├── config.py                    # Configuration settings
│   ├── data/
│   │   ├── __init__.py
│   │   └── preprocessing.py         # Data loading & preprocessing
│   ├── models/
│   │   ├── __init__.py
│   │   └── failure_predictor.py     # Model definitions
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── metrics.py               # Evaluation functions
│   └── utils/
│       ├── __init__.py
│       └── visualization.py         # Plotting utilities
├── scripts/
│   └── train_failure_model.py       # Main training script
├── notebooks/
│   └── MillMate_G17_ENGG2112.ipynb  # Original notebook (reference)
├── data/
│   └── ai4i2020.csv
├── outputs/
│   ├── models/                      # Saved models
│   ├── results/                     # CSV results
│   └── figures/                     # Plots
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Verify Data

Ensure your dataset is at: `data/ai4i2020.csv`

### 3. Run Training Pipeline

```bash
python scripts/train_failure_model.py
```

This will:
- Load and preprocess the data
- Train 6 different model strategies
- Perform 5-fold cross-validation
- Evaluate on test set
- Generate performance visualizations
- Save results to `outputs/`

## 🎯 Model Strategies

The project implements 6 strategies for handling class imbalance:

1. **BalancedRandomForestClassifier** ⭐ (Recommended)
2. **EasyEnsembleClassifier** ⭐ (Recommended)
3. **SMOTE + Logistic Regression**
4. **ADASYN + Logistic Regression**
5. **Class-Weighted Logistic Regression**
6. **Class-Weighted Random Forest**

## 📊 Evaluation Metrics

Models are evaluated using:
- **ROC-AUC**: Overall discriminative ability
- **PR-AUC**: Performance on imbalanced data (primary metric)
- **F1 Score**: Balance of precision and recall
- **MCC**: Matthews Correlation Coefficient
- **Precision & Recall**: At optimal threshold

## ⚙️ Configuration

Modify settings in `src/config.py`:

```python
# Key parameters
TEST_SIZE = 0.2           # Train/test split ratio
CV_FOLDS = 5              # Cross-validation folds
RANDOM_STATE = 42         # Random seed
THRESHOLD_PREFERENCE = 'f1'  # Options: 'f1', 'recall', 'precision'
```

## 📈 Outputs

After running the pipeline, check:

- **Results**: `outputs/results/failure_imbalance_comparison.csv`
- **Figures**: 
  - `outputs/figures/pr_curves_top3.png`
  - `outputs/figures/roc_curves_top3.png`
  - `outputs/figures/model_comparison.png`

## 🔧 Usage Examples

### Run with Custom Configuration

```python
from src import config
from src.data.preprocessing import load_and_preprocess_data
from src.models.failure_predictor import create_failure_predictor

# Modify config as needed
config.TEST_SIZE = 0.3
config.CV_FOLDS = 10

# Run pipeline
X_train, X_test, y_train, y_test, preprocessor = load_and_preprocess_data(config)
predictor = create_failure_predictor(config, preprocessor)
```

### Evaluate a Single Strategy

```python
from src.evaluation.metrics import cross_validate_model, evaluate_on_test

# Get specific model
model = predictor.get_strategy('BalancedRF')

# Cross-validate
cv_results = cross_validate_model(model, X_train, y_train)
print(cv_results)

# Test evaluation
test_metrics, probs, preds = evaluate_on_test(
    model, X_train, y_train, X_test, y_test
)
```

## 🧪 Testing

Run unit tests (when implemented):

```bash
python -m pytest tests/
```

## 📝 Notes

- The original Jupyter notebook is preserved in `notebooks/` for reference
- For production use, consider saving trained models with `joblib` or `pickle`
- Threshold tuning is performed automatically based on `THRESHOLD_PREFERENCE`
- All random operations use fixed seeds for reproducibility

## 👥 Team

MillMate G17 Team - ENGG2112

## 📄 License

[Add your license here]

## 🙏 Acknowledgments

Dataset: AI4I 2020 Predictive Maintenance Dataset