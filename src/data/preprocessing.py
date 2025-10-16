"""
Data loading and preprocessing module for ENG2112 ML Project
Handles data loading, cleaning, encoding, and feature selection
"""
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class DataPreprocessor:
    """Handles all data preprocessing tasks"""
    
    def __init__(self, data_path, target_column, feature_columns, verbose=True):
        """
        Initialize the preprocessor
        
        Args:
            data_path: Path to the CSV file
            target_column: Name of the target column
            feature_columns: List of feature column names
            verbose: Whether to print preprocessing information
        """
        self.data_path = data_path
        self.target_column = target_column
        self.feature_columns = feature_columns
        self.verbose = verbose
        self.df = None
        self.scaler = StandardScaler()
        
    def load_data(self):
        """Load data from CSV file"""
        if self.verbose:
            print(f"Loading data from {self.data_path}...")
        
        self.df = pd.read_csv(self.data_path)
        
        if self.verbose:
            print(f"Initial shape: {self.df.shape}")
            print(f"\nNaN check:\n{self.df.isna().sum()}")
        
        return self
    
    def clean_data(self):
        """Remove NaN values and handle missing data"""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        initial_rows = len(self.df)
        self.df.dropna(inplace=True)
        
        if self.verbose:
            removed = initial_rows - len(self.df)
            print(f"\nRemoved {removed} rows with NaN values")
            print(f"Shape after cleaning: {self.df.shape}")
        
        return self
    
    def encode_type_column(self):
        """Encode product quality labels (L, M, H) to numerical values"""
        if 'Type' in self.df.columns:
            self.df['Type'] = self.df['Type'].map({'L': 0, 'M': 1, 'H': 2})
            if self.verbose:
                print("\nEncoded 'Type' column: L=0, M=1, H=2")
        
        return self
    
    def prepare_features_and_target(self):
        """
        Extract features and target, check target distribution
        
        Returns:
            X: Feature DataFrame
            y: Target array
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Check target exists
        assert self.target_column in self.df.columns, \
            f"Missing '{self.target_column}' column"
        
        # Extract target
        y = self.df[self.target_column].astype(int).values
        
        # Extract features
        X = self.df[self.feature_columns].copy()
        
        if self.verbose:
            print(f"\nTarget: {self.target_column}")
            print(f"Class distribution: {Counter(y)}")
            class_pct = (np.sum(y) / len(y)) * 100
            print(f"Positive class: {class_pct:.2f}%")
            print(f"\nFeatures: {self.feature_columns}")
            print(f"Feature matrix shape: {X.shape}")
        
        return X, y
    
    def split_data(self, X, y, test_size=0.2, stratify=True, random_state=42):
        """
        Split data into train and test sets
        
        Args:
            X: Features
            y: Target
            test_size: Proportion of test set
            stratify: Whether to stratify split by target
            random_state: Random seed
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        stratify_param = y if stratify else None
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            stratify=stratify_param, 
            random_state=random_state
        )
        
        if self.verbose:
            print(f"\nTrain set: {X_train.shape[0]} samples")
            print(f"Test set: {X_test.shape[0]} samples")
            print(f"Train class distribution: {Counter(y_train)}")
            print(f"Test class distribution: {Counter(y_test)}")
        
        return X_train, X_test, y_train, y_test
    
    def get_scaler(self):
        """Return the StandardScaler instance for use in pipelines"""
        return StandardScaler()


def load_and_preprocess_data(config):
    """
    Convenience function to load and preprocess data using config
    
    Args:
        config: Configuration module with required attributes
        
    Returns:
        X_train, X_test, y_train, y_test, preprocessor
    """
    preprocessor = DataPreprocessor(
        data_path=config.DATA_FILE,
        target_column=config.TARGET_COLUMN,
        feature_columns=config.SENSOR_FEATURES,
        verbose=config.VERBOSE
    )
    
    # Execute preprocessing pipeline
    preprocessor.load_data() \
                .clean_data() \
                .encode_type_column()
    
    X, y = preprocessor.prepare_features_and_target()
    
    X_train, X_test, y_train, y_test = preprocessor.split_data(
        X, y,
        test_size=config.TEST_SIZE,
        stratify=config.STRATIFY,
        random_state=config.RANDOM_STATE
    )
    
    return X_train, X_test, y_train, y_test, preprocessor