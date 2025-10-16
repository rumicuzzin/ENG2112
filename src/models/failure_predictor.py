"""
Machine Failure Prediction Models
Defines and manages multiple strategies for handling imbalanced classification
"""
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.ensemble import BalancedRandomForestClassifier, EasyEnsembleClassifier


class FailurePredictor:
    """Manages multiple model strategies for failure prediction"""
    
    def __init__(self, scaler, model_params):
        """
        Initialize failure predictor with strategies
        
        Args:
            scaler: StandardScaler instance for preprocessing
            model_params: Dictionary of model hyperparameters from config
        """
        self.scaler = scaler
        self.model_params = model_params
        self.strategies = self._build_strategies()
    
    def _build_strategies(self):
        """
        Build all model strategies with their pipelines
        
        Returns:
            Dictionary of strategy names to pipeline objects
        """
        params = self.model_params
        
        strategies = {
            # Ensemble methods (recommended for extreme imbalance)
            'BalancedRF': ImbPipeline([
                ('sc', self.scaler),
                ('clf', BalancedRandomForestClassifier(**params['BalancedRF']))
            ]),
            
            'EasyEnsemble': ImbPipeline([
                ('sc', self.scaler),
                ('clf', EasyEnsembleClassifier(**params['EasyEnsemble']))
            ]),
            
            # Oversampling + Linear Model
            'SMOTE + LogReg': ImbPipeline([
                ('sc', self.scaler),
                ('smote', SMOTE(random_state=params['SMOTE_LogReg']['smote_random_state'])),
                ('clf', LogisticRegression(
                    max_iter=params['SMOTE_LogReg']['max_iter'],
                    class_weight=params['SMOTE_LogReg']['class_weight']
                ))
            ]),
            
            'ADASYN + LogReg': ImbPipeline([
                ('sc', self.scaler),
                ('adasyn', ADASYN(random_state=params['ADASYN_LogReg']['adasyn_random_state'])),
                ('clf', LogisticRegression(
                    max_iter=params['ADASYN_LogReg']['max_iter'],
                    class_weight=params['ADASYN_LogReg']['class_weight']
                ))
            ]),
            
            # Class weighting only (no resampling)
            'ClassWeight LogReg': ImbPipeline([
                ('sc', self.scaler),
                ('clf', LogisticRegression(**params['ClassWeight_LogReg']))
            ]),
            
            'ClassWeight RF': ImbPipeline([
                ('sc', self.scaler),
                ('clf', RandomForestClassifier(**params['ClassWeight_RF']))
            ]),
        }
        
        return strategies
    
    def get_strategy(self, name):
        """
        Get a specific strategy by name
        
        Args:
            name: Strategy name
            
        Returns:
            Pipeline object
        """
        if name not in self.strategies:
            raise ValueError(f"Strategy '{name}' not found. Available: {list(self.strategies.keys())}")
        return self.strategies[name]
    
    def get_all_strategies(self):
        """Return all strategies"""
        return self.strategies
    
    def list_strategies(self):
        """Print all available strategies"""
        print("Available strategies:")
        for i, name in enumerate(self.strategies.keys(), 1):
            print(f"  {i}. {name}")


def create_failure_predictor(config, preprocessor):
    """
    Factory function to create FailurePredictor from config
    
    Args:
        config: Configuration module
        preprocessor: DataPreprocessor instance
        
    Returns:
        FailurePredictor instance
    """
    scaler = preprocessor.get_scaler()
    return FailurePredictor(scaler, config.MODEL_PARAMS)