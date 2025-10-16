#!/usr/bin/env python3
"""
Main training script for machine failure prediction
Runs complete pipeline: data loading -> training -> evaluation -> visualization
"""
import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src import config
from src.data.preprocessing import load_and_preprocess_data
from src.models.failure_predictor import create_failure_predictor
from src.evaluation.metrics import evaluate_all_strategies, get_top_models
from src.utils.visualization import plot_pr_curves, plot_roc_curves, plot_model_comparison


def main():
    """Main execution function"""
    
    print("="*70)
    print("ENG2112 - Machine Failure Prediction Training Pipeline")
    print("="*70)
    
    # Step 1: Load and preprocess data
    print("\n[1/5] Loading and preprocessing data...")
    X_train, X_test, y_train, y_test, preprocessor = load_and_preprocess_data(config)
    
    # Step 2: Create failure predictor with all strategies
    print("\n[2/5] Building model strategies...")
    predictor = create_failure_predictor(config, preprocessor)
    predictor.list_strategies()
    
    # Step 3: Evaluate all strategies
    print("\n[3/5] Training and evaluating models...")
    print("This may take several minutes...")
    
    results_df, probs_dict = evaluate_all_strategies(
        strategies=predictor.get_all_strategies(),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        cv_folds=config.CV_FOLDS,
        threshold_preference=config.THRESHOLD_PREFERENCE,
        random_state=config.RANDOM_STATE,
        verbose=config.VERBOSE
    )
    
    # Display results
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print(results_df.to_string(index=False))
    
    # Step 4: Save results
    print(f"\n[4/5] Saving results to {config.RESULTS_CSV}...")
    results_df.to_csv(config.RESULTS_CSV, index=False)
    print("Results saved successfully!")
    
    # Step 5: Generate visualizations
    print(f"\n[5/5] Generating visualizations...")
    top_models = get_top_models(results_df, n=config.TOP_N_MODELS)
    
    print(f"\nTop {config.TOP_N_MODELS} models: {', '.join(top_models)}")
    
    # PR curves
    plot_pr_curves(
        y_test, 
        probs_dict, 
        top_models, 
        save_path=config.PR_CURVE_PLOT
    )
    
    # ROC curves
    plot_roc_curves(
        y_test, 
        probs_dict, 
        top_models, 
        save_path=config.ROC_CURVE_PLOT
    )
    
    # Model comparison
    comparison_path = config.FIGURES_DIR / "model_comparison.png"
    plot_model_comparison(
        results_df,
        metrics=['test_PR_AUC', 'test_F1', 'test_MCC'],
        save_path=comparison_path
    )
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"\nOutputs saved to:")
    print(f"  - Results CSV: {config.RESULTS_CSV}")
    print(f"  - Figures: {config.FIGURES_DIR}/")
    print("\nBest model: " + results_df.iloc[0]['model'])
    print(f"  Test PR-AUC: {results_df.iloc[0]['test_PR_AUC']:.4f}")
    print(f"  Test F1: {results_df.iloc[0]['test_F1']:.4f}")
    print(f"  Test MCC: {results_df.iloc[0]['test_MCC']:.4f}")


if __name__ == "__main__":
    main()