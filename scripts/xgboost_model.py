"""
XGBoost Analysis: Relationship between Type and Failure
Analyzes how product quality type relates to machine failure using sensor data
Uses existing DataPreprocessor for data loading and cleaning
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# Import existing preprocessor
import sys
sys.path.append('src')  # Add src to path
from data.preprocessing import DataPreprocessor


class TypeFailureAnalyzer:
    """Analyzes relationship between Type and Failure using XGBoost"""
    
    def __init__(self, data_path, target='Machine failure', 
                 sensor_features=None, random_state=42):
        """
        Initialize analyzer
        
        Args:
            data_path: Path to CSV data file
            target: Target column name (default: 'Machine failure')
            sensor_features: List of sensor feature names
            random_state: Random seed for reproducibility
        """
        self.data_path = data_path
        self.target = target
        self.sensor_features = sensor_features
        self.random_state = random_state
        self.preprocessor = None
        self.df = None
        self.df_clean = None
        self.models = {}
        self.results = {}
        
    def load_and_prepare_data(self):
        """Load data using existing DataPreprocessor"""
        print("Loading data using DataPreprocessor...")
        
        # Initialize preprocessor
        self.preprocessor = DataPreprocessor(
            data_path=self.data_path,
            target_column=self.target,
            feature_columns=self.sensor_features,
            verbose=True
        )
        
        # Load, clean, and encode
        self.preprocessor.load_data().clean_data().encode_type_column()
        
        # Store the cleaned dataframe
        self.df = self.preprocessor.df.copy()
        
        # Sanitize column names for XGBoost (remove [, ], <, >)
        self.df.columns = self.df.columns.str.replace('[', '_', regex=False)
        self.df.columns = self.df.columns.str.replace(']', '', regex=False)
        self.df.columns = self.df.columns.str.replace('<', '_lt_', regex=False)
        self.df.columns = self.df.columns.str.replace('>', '_gt_', regex=False)
        
        # Update sensor features with sanitized names
        self.sensor_features = [
            f.replace('[', '_').replace(']', '').replace('<', '_lt_').replace('>', '_gt_')
            for f in self.sensor_features
        ]
        
        print(f"\nSanitized sensor features: {self.sensor_features}")
        print(f"\nType distribution:\n{self.df['Type'].value_counts()}")
        
        return self
    
    def analyze_type_failure_correlation(self):
        """Analyze basic correlation between Type and Failure"""
        print("\n" + "="*60)
        print("DIRECT TYPE-FAILURE RELATIONSHIP")
        print("="*60)
        
        # Cross-tabulation
        ct = pd.crosstab(self.df['Type'], self.df[self.target], 
                        normalize='index') * 100
        print("\nFailure rate by Type (%):")
        print(ct)
        
        # Statistical test
        from scipy.stats import chi2_contingency
        contingency_table = pd.crosstab(self.df['Type'], self.df[self.target])
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
        print(f"\nChi-square test:")
        print(f"  χ² = {chi2:.4f}")
        print(f"  p-value = {p_value:.4e}")
        print(f"  Significant relationship: {'Yes' if p_value < 0.05 else 'No'}")
        
        self.results['type_failure_crosstab'] = ct
        self.results['chi2_test'] = {'chi2': chi2, 'p_value': p_value}
        
        return self
    
    def train_model_without_type(self):
        """Train XGBoost model WITHOUT Type feature"""
        print("\n" + "="*60)
        print("MODEL 1: Predicting Failure WITHOUT Type")
        print("="*60)
        
        X = self.df[self.sensor_features].copy()
        y = self.df[self.target].values
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=self.random_state
        )
        
        # Use preprocessor's scaler
        scaler = self.preprocessor.get_scaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train XGBoost
        model = XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            scale_pos_weight=len(y_train[y_train==0]) / len(y_train[y_train==1]),
            random_state=self.random_state,
            eval_metric='logloss'
        )
        
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        print("\nPerformance without Type:")
        print(classification_report(y_test, y_pred, target_names=['No Failure', 'Failure']))
        print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")
        
        self.models['without_type'] = model
        self.results['without_type'] = {
            'y_test': y_test,
            'y_pred': y_pred,
            'y_proba': y_proba,
            'roc_auc': roc_auc_score(y_test, y_proba),
            'X_test': X_test,
            'X_test_scaled': X_test_scaled
        }
        
        return self
    
    def train_model_with_type(self):
        """Train XGBoost model WITH Type feature"""
        print("\n" + "="*60)
        print("MODEL 2: Predicting Failure WITH Type")
        print("="*60)
        
        features_with_type = self.sensor_features + ['Type']
        X = self.df[features_with_type].copy()
        y = self.df[self.target].values
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=self.random_state
        )
        
        # Scale only sensor features (not Type)
        scaler = self.preprocessor.get_scaler()
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        X_train_scaled[self.sensor_features] = scaler.fit_transform(
            X_train[self.sensor_features]
        )
        X_test_scaled[self.sensor_features] = scaler.transform(
            X_test[self.sensor_features]
        )
        
        # Train XGBoost
        model = XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            scale_pos_weight=len(y_train[y_train==0]) / len(y_train[y_train==1]),
            random_state=self.random_state,
            eval_metric='logloss'
        )
        
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        print("\nPerformance with Type:")
        print(classification_report(y_test, y_pred, target_names=['No Failure', 'Failure']))
        print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")
        
        self.models['with_type'] = model
        self.results['with_type'] = {
            'y_test': y_test,
            'y_pred': y_pred,
            'y_proba': y_proba,
            'roc_auc': roc_auc_score(y_test, y_proba),
            'X_test': X_test,
            'X_test_scaled': X_test_scaled,
            'features': features_with_type
        }
        
        return self
    
    def compare_models(self):
        """Compare performance with and without Type"""
        print("\n" + "="*60)
        print("MODEL COMPARISON")
        print("="*60)
        
        roc_without = self.results['without_type']['roc_auc']
        roc_with = self.results['with_type']['roc_auc']
        
        improvement = ((roc_with - roc_without) / roc_without) * 100
        
        print(f"\nROC-AUC without Type: {roc_without:.4f}")
        print(f"ROC-AUC with Type:    {roc_with:.4f}")
        print(f"Improvement:          {improvement:+.2f}%")
        
        if abs(improvement) < 1:
            print("\n→ Type has MINIMAL impact on prediction")
        elif improvement > 0:
            print("\n→ Type IMPROVES prediction (has predictive value)")
        else:
            print("\n→ Type WORSENS prediction (may introduce noise)")
        
        return self
    
    def analyze_feature_importance(self):
        """Analyze feature importance including Type"""
        print("\n" + "="*60)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("="*60)
        
        model = self.models['with_type']
        features = self.results['with_type']['features']
        
        # Get feature importance
        importance = model.feature_importances_
        feature_importance = pd.DataFrame({
            'feature': features,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        print("\nFeature Importance (XGBoost gain):")
        print(feature_importance.to_string(index=False))
        
        type_importance = feature_importance[
            feature_importance['feature'] == 'Type'
        ]['importance'].values[0]
        type_rank = (feature_importance['feature'] == 'Type').values.argmax() + 1
        
        print(f"\nType ranking: #{type_rank} out of {len(features)} features")
        print(f"Type importance: {type_importance:.4f}")
        
        self.results['feature_importance'] = feature_importance
        
        return self
    
    def visualize_results(self, save_path='outputs/figures/type_failure_analysis.png'):
        """Create visualization of results"""
        print("\n" + "="*60)
        print("GENERATING VISUALIZATIONS")
        print("="*60)
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # 1. Failure rate by Type
            print("Creating failure rate chart...")
            ax1 = axes[0, 0]
            ct = self.results['type_failure_crosstab']
            ct.plot(kind='bar', ax=ax1, color=['#2E86AB', '#A23B72'])
            ax1.set_title('Failure Rate by Type', fontsize=12, fontweight='bold')
            ax1.set_xlabel('Type (0=L, 1=M, 2=H)')
            ax1.set_ylabel('Percentage (%)')
            ax1.legend(['No Failure', 'Failure'])
            ax1.grid(True, alpha=0.3)
            ax1.set_xticklabels(ax1.get_xticklabels(), rotation=0)
            
            # 2. Model comparison
            print("Creating model comparison chart...")
            ax2 = axes[0, 1]
            models = ['Without Type', 'With Type']
            roc_scores = [
                self.results['without_type']['roc_auc'],
                self.results['with_type']['roc_auc']
            ]
            bars = ax2.bar(models, roc_scores, color=['#2E86AB', '#F18F01'])
            ax2.set_title('Model Performance Comparison', fontsize=12, fontweight='bold')
            ax2.set_ylabel('ROC-AUC Score')
            ax2.set_ylim([0.8, 1.0])
            ax2.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.4f}', ha='center', va='bottom')
            
            # 3. Feature importance
            print("Creating feature importance chart...")
            ax3 = axes[1, 0]
            fi = self.results['feature_importance'].head(10)
            colors = ['#F18F01' if feat == 'Type' else '#A23B72' for feat in fi['feature']]
            ax3.barh(range(len(fi)), fi['importance'], color=colors)
            ax3.set_yticks(range(len(fi)))
            ax3.set_yticklabels(fi['feature'])
            ax3.invert_yaxis()
            ax3.set_title('Top 10 Feature Importance (Type highlighted)', 
                         fontsize=12, fontweight='bold')
            ax3.set_xlabel('Importance')
            ax3.grid(True, alpha=0.3, axis='x')
            
            # 4. Confusion matrix comparison
            print("Creating confusion matrix...")
            ax4 = axes[1, 1]
            cm_with = confusion_matrix(
                self.results['with_type']['y_test'],
                self.results['with_type']['y_pred']
            )
            sns.heatmap(cm_with, annot=True, fmt='d', cmap='Blues', ax=ax4,
                       xticklabels=['No Failure', 'Failure'],
                       yticklabels=['No Failure', 'Failure'])
            ax4.set_title('Confusion Matrix (With Type)', 
                         fontsize=12, fontweight='bold')
            ax4.set_ylabel('Actual')
            ax4.set_xlabel('Predicted')
            
            plt.tight_layout()
            
            # Save figure
            if save_path:
                import os
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"\n✓ Visualization saved to: {save_path}")
            
            # Try to show, but don't hang if display not available
            try:
                plt.show(block=False)
                plt.pause(0.1)
                print("✓ Visualization displayed")
            except:
                print("✓ Visualization created (display not available)")
            
            plt.close()
            
        except Exception as e:
            print(f"Warning: Could not create visualization: {e}")
            print("Continuing with summary...")
        
        return self
    
    def generate_summary(self):
        """Generate comprehensive summary of findings"""
        print("\n" + "="*60)
        print("SUMMARY: Type-Failure Relationship Analysis")
        print("="*60)
        
        chi2_result = self.results['chi2_test']
        roc_improvement = (
            (self.results['with_type']['roc_auc'] - 
             self.results['without_type']['roc_auc']) / 
            self.results['without_type']['roc_auc']
        ) * 100
        
        type_importance = self.results['feature_importance'][
            self.results['feature_importance']['feature'] == 'Type'
        ]['importance'].values[0]
        
        type_rank = (self.results['feature_importance']['feature'] == 'Type').values.argmax() + 1
        total_features = len(self.results['feature_importance'])
        
        print("\n1. STATISTICAL RELATIONSHIP:")
        print(f"   Chi-square p-value: {chi2_result['p_value']:.4e}")
        print(f"   Statistically significant: {'Yes' if chi2_result['p_value'] < 0.05 else 'No'}")
        
        print("\n2. PREDICTIVE VALUE:")
        print(f"   Performance improvement: {roc_improvement:+.2f}%")
        print(f"   Type feature importance: {type_importance:.4f}")
        print(f"   Type feature rank: #{type_rank} out of {total_features}")
        
        print("\n3. INTERPRETATION:")
        if chi2_result['p_value'] < 0.05:
            print("   ✓ Type HAS a statistically significant relationship with Failure")
            print(f"     (Low-quality products have {self.results['type_failure_crosstab'].iloc[0, 1]:.2f}% failure rate)")
            print(f"     (High-quality products have {self.results['type_failure_crosstab'].iloc[2, 1]:.2f}% failure rate)")
        else:
            print("   ✗ Type has NO statistically significant relationship with Failure")
        
        print("\n4. PREDICTIVE MODELING CONCLUSION:")
        if abs(roc_improvement) < 1 and type_rank > 3:
            print("   → Type adds MINIMAL predictive value")
            print("   → The relationship is likely INDIRECT (mediated by sensor features)")
            print("   → Recommendation: Type can be excluded from failure prediction models")
        elif roc_improvement > 1:
            print("   → Type IMPROVES prediction accuracy")
            print("   → Type captures information NOT fully explained by sensors")
            print("   → Recommendation: Include Type in failure prediction models")
        else:
            print("   → Type has MARGINAL predictive value")
            print("   → Consider domain knowledge before including Type")
        
        return self


def main():
    """Main execution function"""
    # Configuration
    DATA_PATH = 'data/ai4i2020.csv'  # Update with your file path
    SENSOR_FEATURES = [
        'Air temperature [K]',
        'Process temperature [K]',
        'Rotational speed [rpm]',
        'Torque [Nm]',
        'Tool wear [min]'
    ]
    
    # Initialize analyzer
    analyzer = TypeFailureAnalyzer(
        data_path=DATA_PATH,
        target='Machine failure',
        sensor_features=SENSOR_FEATURES,
        random_state=42
    )
    
    # Run complete analysis
    (analyzer
     .load_and_prepare_data()
     .analyze_type_failure_correlation()
     .train_model_without_type()
     .train_model_with_type()
     .compare_models()
     .analyze_feature_importance()
     .visualize_results()
     .generate_summary())
    
    return analyzer


if __name__ == "__main__":
    analyzer = main()
    print("\n" + "="*60)
    print("Analysis complete!")
    print("="*60)