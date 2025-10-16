"""
Visualization utilities for model performance
Creates PR curves, ROC curves, and other performance visualizations
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve


def plot_pr_curves(y_test, probs_dict, model_names, save_path=None, figsize=(8, 6)):
    """
    Plot Precision-Recall curves for multiple models
    
    Args:
        y_test: True labels
        probs_dict: Dictionary mapping model names to predicted probabilities
        model_names: List of model names to plot
        save_path: Path to save figure (optional)
        figsize: Figure size tuple
    """
    plt.figure(figsize=figsize)
    
    for name in model_names:
        if name not in probs_dict:
            print(f"Warning: {name} not found in predictions")
            continue
        
        prob = probs_dict[name]
        precision, recall, _ = precision_recall_curve(y_test, prob)
        
        # Calculate PR-AUC for legend
        from sklearn.metrics import average_precision_score
        pr_auc = average_precision_score(y_test, prob)
        
        plt.plot(recall, precision, label=f'{name} (PR-AUC={pr_auc:.3f})', linewidth=2)
    
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curves (Top Models)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best', fontsize=10)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved PR curve to: {save_path}")
    
    plt.show()


def plot_roc_curves(y_test, probs_dict, model_names, save_path=None, figsize=(8, 6)):
    """
    Plot ROC curves for multiple models
    
    Args:
        y_test: True labels
        probs_dict: Dictionary mapping model names to predicted probabilities
        model_names: List of model names to plot
        save_path: Path to save figure (optional)
        figsize: Figure size tuple
    """
    plt.figure(figsize=figsize)
    
    for name in model_names:
        if name not in probs_dict:
            print(f"Warning: {name} not found in predictions")
            continue
        
        prob = probs_dict[name]
        fpr, tpr, _ = roc_curve(y_test, prob)
        
        # Calculate ROC-AUC for legend
        from sklearn.metrics import roc_auc_score
        roc_auc = roc_auc_score(y_test, prob)
        
        plt.plot(fpr, tpr, label=f'{name} (ROC-AUC={roc_auc:.3f})', linewidth=2)
    
    # Plot diagonal reference line
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves (Top Models)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best', fontsize=10)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved ROC curve to: {save_path}")
    
    plt.show()


def plot_model_comparison(results_df, metrics=['test_PR_AUC', 'test_F1', 'test_MCC'], 
                         save_path=None, figsize=(12, 6)):
    """
    Create bar chart comparing models across multiple metrics
    
    Args:
        results_df: Results DataFrame
        metrics: List of metric columns to compare
        save_path: Path to save figure (optional)
        figsize: Figure size tuple
    """
    n_metrics = len(metrics)
    x = np.arange(len(results_df))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    
    for i, metric in enumerate(metrics):
        offset = width * (i - n_metrics/2 + 0.5)
        metric_name = metric.replace('test_', '').replace('_', '-')
        ax.bar(x + offset, results_df[metric], width, 
               label=metric_name, color=colors[i % len(colors)])
    
    ax.set_xlabel('Models', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(results_df['model'], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved comparison chart to: {save_path}")
    
    plt.show()


def display_confusion_matrices(results_df, n_cols=3, figsize=(15, 10)):
    """
    Display confusion matrices for all models in a grid
    
    Args:
        results_df: Results DataFrame with conf_mat column
        n_cols: Number of columns in grid
        figsize: Figure size tuple
    """
    n_models = len(results_df)
    n_rows = (n_models + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_models > 1 else [axes]
    
    for idx, (_, row) in enumerate(results_df.iterrows()):
        ax = axes[idx]
        cm = row['test_conf_mat']
        
        # Plot confusion matrix
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.set_title(row['model'], fontsize=10, fontweight='bold')
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black")
        
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['No Fail', 'Fail'])
        ax.set_yticklabels(['No Fail', 'Fail'])
    
    # Hide unused subplots
    for idx in range(n_models, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.show()