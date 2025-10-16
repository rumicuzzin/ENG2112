"""
Evaluation metrics and model assessment functions
Handles cross-validation, test set evaluation, and threshold tuning
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    average_precision_score, roc_auc_score, precision_recall_curve,
    confusion_matrix, matthews_corrcoef, f1_score, 
    precision_score, recall_score
)


def cross_validate_model(model, X, y, cv_folds=5, random_state=42, verbose=False):
    """
    Perform stratified K-fold cross-validation
    
    Args:
        model: sklearn/imblearn pipeline
        X: Feature DataFrame
        y: Target array
        cv_folds: Number of CV folds
        random_state: Random seed
        verbose: Print fold-by-fold results
        
    Returns:
        Dictionary with mean ROC-AUC, PR-AUC, F1, and MCC
    """
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    roc_list, pr_list, f1_list, mcc_list = [], [], [], []
    
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
        # Fit and predict
        model.fit(X.iloc[tr_idx], y[tr_idx])
        prob = model.predict_proba(X.iloc[va_idx])[:, 1]
        
        # Find optimal threshold for F1
        P, R, T = precision_recall_curve(y[va_idx], prob)
        f1_scores = 2 * P * R / (P + R + 1e-12)
        best_threshold = T[np.nanargmax(f1_scores)] if len(T) > 0 else 0.5
        pred = (prob >= best_threshold).astype(int)
        
        # Calculate metrics
        roc_auc = roc_auc_score(y[va_idx], prob)
        pr_auc = average_precision_score(y[va_idx], prob)
        f1 = f1_score(y[va_idx], pred)
        mcc = matthews_corrcoef(y[va_idx], pred)
        
        roc_list.append(roc_auc)
        pr_list.append(pr_auc)
        f1_list.append(f1)
        mcc_list.append(mcc)
        
        if verbose:
            print(f"Fold {fold}: ROC-AUC={roc_auc:.4f}, PR-AUC={pr_auc:.4f}, "
                  f"F1={f1:.4f}, MCC={mcc:.4f}")
    
    return {
        'cv_ROC_AUC': np.mean(roc_list),
        'cv_PR_AUC': np.mean(pr_list),
        'cv_F1': np.mean(f1_list),
        'cv_MCC': np.mean(mcc_list)
    }


def evaluate_on_test(model, X_train, y_train, X_test, y_test, 
                     threshold_preference='f1', verbose=False):
    """
    Train on full training set and evaluate on test set with threshold tuning
    
    Args:
        model: sklearn/imblearn pipeline
        X_train, y_train: Training data
        X_test, y_test: Test data
        threshold_preference: 'f1', 'recall', or 'precision'
        verbose: Print detailed results
        
    Returns:
        metrics: Dictionary of test metrics
        prob: Predicted probabilities
        pred: Binary predictions
    """
    # Fit model
    model.fit(X_train, y_train)
    
    # Predict probabilities
    prob = model.predict_proba(X_test)[:, 1]
    
    # Threshold tuning
    P, R, T = precision_recall_curve(y_test, prob)
    T = np.r_[T, 1.0]  # Add final threshold
    
    if threshold_preference == 'recall':
        idx = np.argmax(R)
    elif threshold_preference == 'precision':
        idx = np.argmax(P)
    else:  # default: f1
        f1_scores = 2 * P * R / (P + R + 1e-12)
        idx = np.nanargmax(f1_scores)
    
    optimal_threshold = float(T[idx])
    pred = (prob >= optimal_threshold).astype(int)
    
    # Calculate all metrics
    metrics = {
        'ROC_AUC': roc_auc_score(y_test, prob),
        'PR_AUC': average_precision_score(y_test, prob),
        'F1': f1_score(y_test, pred),
        'Recall': recall_score(y_test, pred),
        'Precision': precision_score(y_test, pred),
        'MCC': matthews_corrcoef(y_test, pred),
        'threshold': optimal_threshold,
        'conf_mat': confusion_matrix(y_test, pred)
    }
    
    if verbose:
        print(f"\nTest Set Metrics (threshold={optimal_threshold:.4f}):")
        for k, v in metrics.items():
            if k != 'conf_mat':
                print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
        print(f"\nConfusion Matrix:\n{metrics['conf_mat']}")
    
    return metrics, prob, pred


def evaluate_all_strategies(strategies, X_train, y_train, X_test, y_test, 
                            cv_folds=5, threshold_preference='f1', 
                            random_state=42, verbose=True):
    """
    Evaluate all strategies with CV and test set performance
    
    Args:
        strategies: Dictionary of model pipelines
        X_train, y_train: Training data
        X_test, y_test: Test data
        cv_folds: Number of CV folds
        threshold_preference: Threshold optimization criterion
        random_state: Random seed
        verbose: Print results for each model
        
    Returns:
        results_df: DataFrame with all results
        probs_dict: Dictionary of test probabilities for each model
    """
    rows = []
    probs_dict = {}
    
    for name, pipeline in strategies.items():
        if verbose:
            print(f"\n{'='*60}")
            print(f"Evaluating: {name}")
            print('='*60)
        
        # Cross-validation
        cv_metrics = cross_validate_model(
            pipeline, X_train, y_train, 
            cv_folds=cv_folds, 
            random_state=random_state,
            verbose=False
        )
        
        # Test set evaluation
        test_metrics, prob, pred = evaluate_on_test(
            pipeline, X_train, y_train, X_test, y_test,
            threshold_preference=threshold_preference,
            verbose=verbose
        )
        
        # Combine results
        row = {'model': name}
        row.update(cv_metrics)
        row.update({f'test_{k}': v for k, v in test_metrics.items() if k != 'conf_mat'})
        rows.append(row)
        
        probs_dict[name] = prob
        
        if verbose:
            print(f"\nSummary:")
            print(f"  CV PR-AUC: {cv_metrics['cv_PR_AUC']:.4f}")
            print(f"  Test PR-AUC: {test_metrics['PR_AUC']:.4f}")
            print(f"  Test F1: {test_metrics['F1']:.4f}")
            print(f"  Test MCC: {test_metrics['MCC']:.4f}")
    
    # Create results DataFrame sorted by performance
    results_df = pd.DataFrame(rows).sort_values(
        by=['test_PR_AUC', 'test_MCC', 'test_F1'], 
        ascending=False
    )
    
    return results_df, probs_dict


def get_top_models(results_df, n=3):
    """
    Get names of top N models by performance
    
    Args:
        results_df: Results DataFrame
        n: Number of top models
        
    Returns:
        List of top model names
    """
    return results_df.head(n)['model'].tolist()