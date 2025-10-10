# MillMate G17 - Predictive Maintenance ML Pipeline

A machine learning project for predictive maintenance using the AI4I 2020 dataset. This notebook implements multiple classification strategies to predict product quality and machine failures.

## Project Overview

This project tackles two classification problems:

1. **Product Quality Classification** - Predicting product type: Low (L), Medium (M), or High (H) quality
2. **Machine Failure Prediction** - Predicting whether a machine will fail based on sensor readings

## Prerequisites

- Python 3.11 or higher
- VS Code with Python extension
- Jupyter extension for VS Code

## Setup Instructions

### 1. Project Structure

Ensure your directory looks like this:

```
project/
├── README.md
├── requirements.txt
├── MillMate_G17_ENGG2112.ipynb
└── data/
    └── ai4i2020.csv
```

### 2. Create Virtual Environment

**Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run in VS Code

**Option A: Open the notebook directly**
1. Open `MillMate_G17_ENGG2112.ipynb` in VS Code
2. Select your virtual environment kernel (top right)
3. Run All Cells

**Option B: Launch Jupyter from terminal**
```bash
jupyter lab
```
Then open the notebook in your browser at `http://localhost:8888`

## Dataset

Place your dataset file at: `data/ai4i2020.csv`

**Features:**
- Air temperature [K]
- Process temperature [K]
- Rotational speed [rpm]
- Torque [Nm]
- Tool wear [min]

## Dependencies

The `requirements.txt` file includes:

```
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
imbalanced-learn>=0.11.0
matplotlib>=3.7.0
jupyterlab>=4.0.0
ipykernel>=6.25.0
```

## Notebook Structure

### Cell 1: Data Preprocessing
- Loads the dataset
- Checks for missing values
- Encodes product quality labels
- Applies Z-score normalization

### Cell 2: Multiclass Quality Prediction
- Implements 5 strategies for handling imbalanced classes
- Compares models using cross-validation
- **Known Issue:** Error with `average_precision_score` for multiclass - needs fixing

### Cell 3: Binary Machine Failure Prediction
- Predicts machine failures (3.4% failure rate)
- Tests 6 different imbalance handling strategies
- Outputs comparison table and visualizations

## Output Files

After running the notebook:
- `failure_imbalance_comparison.csv` - Model performance comparison

## Troubleshooting

### File Not Found Error

If you get a file not found error, update the path in the notebook:

```python
df = pd.read_csv("data/ai4i2020.csv")
```

### Kernel Not Found

Make sure your virtual environment is activated and selected as the kernel in VS Code.

### Missing Packages

If imports fail, reinstall dependencies:
```bash
pip install -r requirements.txt
```

## Deactivating Virtual Environment

When finished:
```bash
deactivate
```

## Authors

MillMate G17 Team - ENGG2112