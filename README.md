# ENG2112 - Machine Learning for Predictive Maintenance

A structured Python implementation of machine learning models for predictive maintenance using the AI4I 2020 dataset. This project includes failure prediction models and advanced analysis of machine type classification.

## 📋 Project Overview

**Dataset**: AI4I 2020 Predictive Maintenance Dataset  
**Primary Task**: Binary classification of machine failures (~3.4% failure rate)  
**Secondary Task**: Multi-class classification of machine type (L/M/H)  
**Features**: Sensor readings (temperature, speed, torque, tool wear) + failure modes

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
│   │   └── failure_predictor.py     # Failure prediction models
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── metrics.py               # Evaluation functions
│   └── utils/
│       ├── __init__.py
│       └── visualization.py         # Plotting utilities
├── scripts/
│   ├── train_failure_model.py       # Main training script
│   ├── xgboost_model.py            # Type-Failure relationship analysis
│   └── neural_network_model.py     # Neural network type classifier
├── notebooks/
│   └── MillMate_G17_ENGG2112.ipynb  # Original notebook (reference)
├── data/
│   └── ai4i2020.csv
├── outputs/
│   ├── models/                      # Saved models (.h5, .pkl)
│   ├── results/                     # CSV results
│   └── figures/                     # Plots and visualizations
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

### 3. Run Analysis Pipelines

#### Failure Prediction (Main Task)
```bash
python scripts/train_failure_model.py
```

#### Type-Failure Relationship Analysis
```bash
python scripts/xgboost_model.py
```

#### Type Classification from Conditions
```bash
python scripts/neural_network_model.py
```

## 🎯 Models & Approaches

### 1. Failure Prediction Models

Implements 6 strategies for handling extreme class imbalance:

1. **BalancedRandomForestClassifier** ⭐ (Recommended)
2. **EasyEnsembleClassifier** ⭐ (Recommended)
3. **SMOTE + Logistic Regression**
4. **ADASYN + Logistic Regression**
5. **Class-Weighted Logistic Regression**
6. **Class-Weighted Random Forest**

**Key Features:**
- 5-fold stratified cross-validation
- Threshold tuning for optimal F1/Precision/Recall
- Comprehensive performance metrics
- Handles 97:3 class imbalance

### 2. XGBoost Type-Failure Analysis

Analyzes the relationship between product quality type (L/M/H) and machine failures.

**What it does:**
- Statistical significance testing (Chi-square)
- Comparative modeling (with/without Type feature)
- Feature importance ranking
- Determines if Type has predictive value for failures

**Usage:**
```bash
python scripts/xgboost_model.py
```

**Outputs:**
- Type-failure correlation statistics
- Model performance comparison
- Feature importance visualization
- Figure saved to: `outputs/figures/type_failure_analysis.png`

### 3. Neural Network Type Classifier

Predicts machine type (L/M/H) from sensor conditions and failure patterns.

**What it does:**
- Reverse-engineers quality type from operational data
- Multi-class classification using deep learning
- Handles class imbalance with weighted loss
- Analyzes which features indicate different quality levels

**Architecture:**
- Input: 11 features (5 sensors + 6 failure modes)
- Hidden layers: 256 → 128 → 64 neurons
- Dropout (0.4) and L2 regularization
- Batch normalization for stable training
- Output: 3 classes (Type L, M, H)

**Usage:**
```bash
python scripts/neural_network_model.py
```

**Outputs:**
- Per-class accuracy metrics
- Feature importance analysis
- Training history plots
- Confusion matrices
- Saved model: `outputs/models/nn_type_classifier.h5`
- Figure saved to: `outputs/figures/nn_type_classifier.png`

## 📊 Evaluation Metrics

### Failure Prediction Metrics
- **ROC-AUC**: Overall discriminative ability
- **PR-AUC**: Performance on imbalanced data (primary metric)
- **F1 Score**: Balance of precision and recall
- **MCC**: Matthews Correlation Coefficient
- **Precision & Recall**: At optimal threshold

### Type Classification Metrics
- **Accuracy**: Overall classification accuracy
- **F1-Score (Macro/Weighted)**: Per-class performance
- **Confusion Matrix**: Detailed prediction breakdown
- **Per-Class Recall**: Type-specific accuracy

## ⚙️ Configuration

### Failure Prediction Config (`src/config.py`)
```python
# Key parameters
TEST_SIZE = 0.2           # Train/test split ratio
CV_FOLDS = 5              # Cross-validation folds
RANDOM_STATE = 42         # Random seed
THRESHOLD_PREFERENCE = 'f1'  # Options: 'f1', 'recall', 'precision'
```

### XGBoost Analysis Config (`scripts/xgboost_model.py`)
```python
DATA_PATH = 'data/ai4i2020.csv'
SENSOR_FEATURES = ['Air temperature [K]', 'Process temperature [K]', ...]
random_state = 42
```

### Neural Network Config (`scripts/neural_network_model.py`)
```python
hidden_layers = [256, 128, 64]
dropout_rate = 0.4
learning_rate = 0.001
epochs = 150
batch_size = 64
```

## 📈 Outputs

### Failure Prediction
- **Results**: `outputs/results/failure_imbalance_comparison.csv`
- **Figures**: 
  - `outputs/figures/pr_curves_top3.png`
  - `outputs/figures/roc_curves_top3.png`
  - `outputs/figures/model_comparison.png`

### Type-Failure Analysis
- **Figure**: `outputs/figures/type_failure_analysis.png`
  - Failure rate by type
  - Model comparison (with/without Type)
  - Feature importance
  - Confusion matrix

### Type Classification
- **Model**: `outputs/models/nn_type_classifier.h5`
- **Figure**: `outputs/figures/nn_type_classifier.png`
  - Training/validation curves
  - Confusion matrices
  - Feature importance
  - Prediction confidence distribution

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

### Load and Use Trained Neural Network

```python
from tensorflow import keras
import numpy as np

# Load saved model
model = keras.models.load_model('outputs/models/nn_type_classifier.h5')

# Predict type from new data
predictions = model.predict(new_data_scaled)
predicted_types = np.argmax(predictions, axis=1)  # 0=L, 1=M, 2=H
```

## 🧪 Testing

Run unit tests (when implemented):

```bash
python -m pytest tests/
```

## 📊 Key Findings

### Failure Prediction
- Ensemble methods (BalancedRF, EasyEnsemble) perform best
- PR-AUC > 0.95 achievable on test set
- Threshold tuning critical for practical deployment

### Type-Failure Relationship
- Statistically significant relationship exists (p < 0.001)
- Type L (Low quality): ~3.9% failure rate
- Type H (High quality): ~2.1% failure rate
- However, Type adds minimal predictive value beyond sensor features

### Type Classification
- Neural network can classify type from conditions
- Accuracy depends on feature information content
- Demonstrates reverse-engineering of quality from operational data

## 📝 Notes

- The original Jupyter notebook is preserved in `notebooks/` for reference
- All models use fixed random seeds for reproducibility
- Neural network models are saved in HDF5 format (.h5)
- XGBoost uses non-interactive matplotlib backend for server compatibility
- Class weights automatically computed for imbalanced datasets

## 🔬 Research Questions Addressed

1. **Can we predict machine failures from sensor data?** 
   → Yes, with >95% PR-AUC using ensemble methods

2. **Does product quality type relate to failure risk?**
   → Yes statistically, but relationship is weak and indirect

3. **Can we classify machine type from operational conditions?**
   → Investigated via neural network classifier

## 👥 Team

MillMate G17 Team - ENGG2112

## 📄 License

[Add your license here]

## 🙏 Acknowledgments

Dataset: AI4I 2020 Predictive Maintenance Dataset

## 📚 Requirements

- Python 3.8+
- scikit-learn
- imbalanced-learn
- xgboost
- tensorflow
- pandas
- numpy
- matplotlib
- seaborn
- scipy

See `requirements.txt` for full dependency list.