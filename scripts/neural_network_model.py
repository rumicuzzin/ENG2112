"""
Neural Network Type Classification
Predicts product quality type (L, M, H) based on sensor conditions and failure patterns
Uses existing DataPreprocessor for data loading and cleaning
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    f1_score, precision_score, recall_score
)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
import warnings
warnings.filterwarnings('ignore')

# Import existing preprocessor
import sys
sys.path.append('src')
from data.preprocessing import DataPreprocessor


class TypeClassifierNN:
    """Neural Network classifier for predicting Type from sensor conditions and failures"""
    
    def __init__(self, data_path, target='Type', sensor_features=None, 
                 failure_features=None, random_state=42):
        """
        Initialize classifier
        
        Args:
            data_path: Path to CSV data file
            target: Target column name (default: 'Type')
            sensor_features: List of sensor feature names
            failure_features: List of failure mode features
            random_state: Random seed for reproducibility
        """
        self.data_path = data_path
        self.target = target
        self.sensor_features = sensor_features
        self.failure_features = failure_features
        self.random_state = random_state
        self.preprocessor = None
        self.df = None
        self.scaler = StandardScaler()
        self.model = None
        self.history = None
        self.results = {}
        
        # Set random seeds
        np.random.seed(random_state)
        tf.random.set_seed(random_state)
        
    def load_and_prepare_data(self):
        """Load data using existing DataPreprocessor"""
        print("Loading data using DataPreprocessor...")
        
        # For this task, we need all features (sensors + failures)
        all_features = self.sensor_features + self.failure_features
        
        # Initialize preprocessor
        self.preprocessor = DataPreprocessor(
            data_path=self.data_path,
            target_column='Machine failure',  # Dummy target for preprocessor
            feature_columns=self.sensor_features,
            verbose=True
        )
        
        # Load and clean data
        self.preprocessor.load_data().clean_data()
        
        # Store the cleaned dataframe
        self.df = self.preprocessor.df.copy()
        
        # Encode Type BEFORE making it the target
        if 'Type' in self.df.columns:
            self.df['Type'] = self.df['Type'].map({'L': 0, 'M': 1, 'H': 2})
            print("\nEncoded Type: L=0, M=1, H=2")
        
        print(f"\nType distribution:\n{self.df[self.target].value_counts()}")
        print(f"\nFeatures for prediction:")
        print(f"  Sensor features: {self.sensor_features}")
        print(f"  Failure features: {self.failure_features}")
        
        return self
    
    def prepare_train_test_split(self, test_size=0.2, val_size=0.2):
        """Prepare train, validation, and test sets"""
        print("\n" + "="*60)
        print("PREPARING DATA SPLITS")
        print("="*60)
        
        # Combine all features
        all_features = self.sensor_features + self.failure_features
        
        X = self.df[all_features].copy()
        y = self.df[self.target].values
        
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=self.random_state
        )
        
        # Second split: train vs val
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, 
            stratify=y_temp, random_state=self.random_state
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"\nTrain set: {X_train.shape[0]} samples")
        print(f"Validation set: {X_val.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        print(f"\nTrain distribution: {np.bincount(y_train)}")
        print(f"Val distribution: {np.bincount(y_val)}")
        print(f"Test distribution: {np.bincount(y_test)}")
        
        self.results['X_train'] = X_train_scaled
        self.results['X_val'] = X_val_scaled
        self.results['X_test'] = X_test_scaled
        self.results['y_train'] = y_train
        self.results['y_val'] = y_val
        self.results['y_test'] = y_test
        self.results['feature_names'] = all_features
        
        return self
    
    def build_model(self, hidden_layers=[256, 128, 64], dropout_rate=0.4, 
                    learning_rate=0.001):
        """
        Build neural network architecture
        
        Args:
            hidden_layers: List of neurons in each hidden layer
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate for optimizer
        """
        print("\n" + "="*60)
        print("BUILDING NEURAL NETWORK")
        print("="*60)
        
        n_features = self.results['X_train'].shape[1]
        n_classes = 3  # L, M, H
        
        model = keras.Sequential()
        
        # Input layer
        model.add(layers.Input(shape=(n_features,)))
        
        # Hidden layers with batch normalization and dropout
        for i, units in enumerate(hidden_layers):
            model.add(layers.Dense(units, activation='relu', 
                                  kernel_regularizer=keras.regularizers.l2(0.001),
                                  name=f'hidden_{i+1}'))
            model.add(layers.BatchNormalization())
            model.add(layers.Dropout(dropout_rate))
        
        # Output layer (3 classes)
        model.add(layers.Dense(n_classes, activation='softmax', name='output'))
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        
        print(f"\nArchitecture:")
        print(f"  Input features: {n_features}")
        print(f"  Hidden layers: {hidden_layers}")
        print(f"  Dropout rate: {dropout_rate}")
        print(f"  L2 regularization: 0.001")
        print(f"  Output classes: {n_classes} (L=0, M=1, H=2)")
        print(f"  Learning rate: {learning_rate}")
        print(f"\nModel summary:")
        model.summary()
        
        return self
    
    def train_model(self, epochs=150, batch_size=64, patience=20):
        """
        Train the neural network
        
        Args:
            epochs: Maximum number of training epochs
            batch_size: Batch size for training
            patience: Early stopping patience
        """
        print("\n" + "="*60)
        print("TRAINING NEURAL NETWORK")
        print("="*60)
        
        # Calculate class weights to handle imbalance
        from sklearn.utils.class_weight import compute_class_weight
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(self.results['y_train']),
            y=self.results['y_train']
        )
        class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
        
        print(f"\nClass weights (to handle imbalance):")
        print(f"  Type L (0): {class_weight_dict[0]:.3f}")
        print(f"  Type M (1): {class_weight_dict[1]:.3f}")
        print(f"  Type H (2): {class_weight_dict[2]:.3f}")
        
        # Callbacks
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7,
            min_lr=1e-6,
            verbose=1
        )
        
        # Train model with class weights
        self.history = self.model.fit(
            self.results['X_train'],
            self.results['y_train'],
            validation_data=(self.results['X_val'], self.results['y_val']),
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weight_dict,  # Add class weights
            callbacks=[early_stop, reduce_lr],
            verbose=1
        )
        
        print("\n✓ Training complete!")
        
        return self
    
    def evaluate_model(self):
        """Evaluate model on test set"""
        print("\n" + "="*60)
        print("EVALUATING ON TEST SET")
        print("="*60)
        
        X_test = self.results['X_test']
        y_test = self.results['y_test']
        
        # Predictions
        y_pred_proba = self.model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        
        print(f"\nTest Set Performance:")
        print(f"  Accuracy:           {accuracy:.4f}")
        print(f"  F1-Score (macro):   {f1_macro:.4f}")
        print(f"  F1-Score (weighted): {f1_weighted:.4f}")
        print(f"  Precision:          {precision:.4f}")
        print(f"  Recall:             {recall:.4f}")
        
        print("\n" + "-"*60)
        print("Detailed Classification Report:")
        print("-"*60)
        target_names = ['Type L (Low)', 'Type M (Medium)', 'Type H (High)']
        print(classification_report(y_test, y_pred, target_names=target_names))
        
        # Store results
        self.results['y_pred'] = y_pred
        self.results['y_pred_proba'] = y_pred_proba
        self.results['accuracy'] = accuracy
        self.results['f1_macro'] = f1_macro
        self.results['f1_weighted'] = f1_weighted
        self.results['confusion_matrix'] = confusion_matrix(y_test, y_pred)
        
        return self
    
    def analyze_feature_importance(self):
        """
        Analyze feature importance using permutation importance
        Note: This is computationally expensive for neural networks
        """
        print("\n" + "="*60)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("="*60)
        print("Analyzing which features most influence Type prediction...")
        
        from sklearn.inspection import permutation_importance
        
        # Create a wrapper for the Keras model to work with sklearn
        def model_predict(X):
            return np.argmax(self.model.predict(X, verbose=0), axis=1)
        
        # Custom scoring function for multiclass
        from sklearn.metrics import make_scorer, accuracy_score
        
        try:
            # Calculate permutation importance with custom scorer
            perm_importance = permutation_importance(
                self.model,
                self.results['X_test'],
                self.results['y_test'],
                n_repeats=10,
                random_state=self.random_state,
                scoring=make_scorer(accuracy_score),
                n_jobs=-1
            )
            
            # Create feature importance dataframe
            feature_importance = pd.DataFrame({
                'feature': self.results['feature_names'],
                'importance': perm_importance.importances_mean,
                'std': perm_importance.importances_std
            }).sort_values('importance', ascending=False)
            
            print("\nFeature Importance (Permutation):")
            print(feature_importance.to_string(index=False))
            
            self.results['feature_importance'] = feature_importance
            
        except Exception as e:
            print(f"\nWarning: Could not compute permutation importance: {e}")
            print("Using gradient-based feature importance instead...")
            
            # Alternative: Use gradient-based importance
            self._compute_gradient_importance()
        
        return self
    
    def _compute_gradient_importance(self):
        """Compute feature importance using gradients"""
        X_test = self.results['X_test']
        
        # Convert to tensor
        X_tensor = tf.constant(X_test, dtype=tf.float32)
        
        # Compute gradients for each sample
        all_gradients = []
        for i in range(len(X_test)):
            with tf.GradientTape() as tape:
                tape.watch(X_tensor)
                predictions = self.model(X_tensor[i:i+1])
                target_class = self.results['y_test'][i]
                loss = predictions[0, target_class]
            
            gradients = tape.gradient(loss, X_tensor)
            all_gradients.append(np.abs(gradients.numpy()[i]))
        
        # Average gradients across all samples
        avg_gradients = np.mean(all_gradients, axis=0)
        
        # Create feature importance dataframe
        feature_importance = pd.DataFrame({
            'feature': self.results['feature_names'],
            'importance': avg_gradients
        }).sort_values('importance', ascending=False)
        
        print("\nFeature Importance (Gradient-based):")
        print(feature_importance.to_string(index=False))
        
        self.results['feature_importance'] = feature_importance
    
    def visualize_results(self, save_path='outputs/figures/nn_type_classifier.png'):
        """Create comprehensive visualization of results"""
        print("\n" + "="*60)
        print("GENERATING VISUALIZATIONS")
        print("="*60)
        
        try:
            fig = plt.figure(figsize=(18, 12))
            gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
            
            # 1. Training history - Loss
            print("Creating training history plots...")
            ax1 = fig.add_subplot(gs[0, 0])
            ax1.plot(self.history.history['loss'], label='Train Loss', linewidth=2)
            ax1.plot(self.history.history['val_loss'], label='Val Loss', linewidth=2)
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.set_title('Training and Validation Loss', fontweight='bold')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 2. Training history - Accuracy
            ax2 = fig.add_subplot(gs[0, 1])
            ax2.plot(self.history.history['accuracy'], label='Train Acc', linewidth=2)
            ax2.plot(self.history.history['val_accuracy'], label='Val Acc', linewidth=2)
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy')
            ax2.set_title('Training and Validation Accuracy', fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # 3. Confusion Matrix
            print("Creating confusion matrix...")
            ax3 = fig.add_subplot(gs[0, 2])
            cm = self.results['confusion_matrix']
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3,
                       xticklabels=['L', 'M', 'H'],
                       yticklabels=['L', 'M', 'H'])
            ax3.set_title('Confusion Matrix', fontweight='bold')
            ax3.set_ylabel('True Type')
            ax3.set_xlabel('Predicted Type')
            
            # 4. Per-class accuracy
            ax4 = fig.add_subplot(gs[1, 0])
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            class_acc = cm_normalized.diagonal()
            bars = ax4.bar(['Type L', 'Type M', 'Type H'], class_acc, 
                          color=['#2E86AB', '#A23B72', '#F18F01'])
            ax4.set_ylabel('Accuracy')
            ax4.set_title('Per-Class Accuracy', fontweight='bold')
            ax4.set_ylim([0, 1])
            ax4.grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}', ha='center', va='bottom')
            
            # 5. Feature importance
            print("Creating feature importance plot...")
            ax5 = fig.add_subplot(gs[1, 1:])
            fi = self.results['feature_importance'].head(10)
            ax5.barh(range(len(fi)), fi['importance'], color='#A23B72')
            ax5.set_yticks(range(len(fi)))
            ax5.set_yticklabels(fi['feature'])
            ax5.invert_yaxis()
            ax5.set_xlabel('Importance (Permutation)')
            ax5.set_title('Top 10 Most Important Features', fontweight='bold')
            ax5.grid(True, alpha=0.3, axis='x')
            
            # 6. Prediction confidence distribution
            ax6 = fig.add_subplot(gs[2, 0])
            max_probs = np.max(self.results['y_pred_proba'], axis=1)
            ax6.hist(max_probs, bins=30, color='#2E86AB', alpha=0.7, edgecolor='black')
            ax6.set_xlabel('Prediction Confidence')
            ax6.set_ylabel('Frequency')
            ax6.set_title('Model Confidence Distribution', fontweight='bold')
            ax6.axvline(np.mean(max_probs), color='red', linestyle='--', 
                       linewidth=2, label=f'Mean: {np.mean(max_probs):.3f}')
            ax6.legend()
            ax6.grid(True, alpha=0.3, axis='y')
            
            # 7. Normalized confusion matrix
            ax7 = fig.add_subplot(gs[2, 1])
            sns.heatmap(cm_normalized, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax7,
                       xticklabels=['L', 'M', 'H'],
                       yticklabels=['L', 'M', 'H'],
                       vmin=0, vmax=1)
            ax7.set_title('Normalized Confusion Matrix', fontweight='bold')
            ax7.set_ylabel('True Type')
            ax7.set_xlabel('Predicted Type')
            
            # 8. Metrics summary
            ax8 = fig.add_subplot(gs[2, 2])
            ax8.axis('off')
            metrics_text = f"""
            PERFORMANCE SUMMARY
            
            Overall Accuracy: {self.results['accuracy']:.4f}
            
            F1-Score (Macro): {self.results['f1_macro']:.4f}
            F1-Score (Weighted): {self.results['f1_weighted']:.4f}
            
            Mean Confidence: {np.mean(max_probs):.4f}
            
            Samples per Class:
              Type L: {np.sum(self.results['y_test'] == 0)}
              Type M: {np.sum(self.results['y_test'] == 1)}
              Type H: {np.sum(self.results['y_test'] == 2)}
            
            Total Test Samples: {len(self.results['y_test'])}
            """
            ax8.text(0.1, 0.5, metrics_text, fontsize=11, 
                    verticalalignment='center', family='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
            
            # Save figure
            if save_path:
                import os
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"\n✓ Visualization saved to: {save_path}")
            
            plt.close()
            print("✓ Visualization created successfully")
            
        except Exception as e:
            print(f"Warning: Could not create visualization: {e}")
        
        return self
    
    def generate_summary(self):
        """Generate comprehensive summary of findings"""
        print("\n" + "="*60)
        print("SUMMARY: Type Classification Results")
        print("="*60)
        
        accuracy = self.results['accuracy']
        f1_macro = self.results['f1_macro']
        cm = self.results['confusion_matrix']
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        print(f"\n1. OVERALL PERFORMANCE:")
        print(f"   Test Accuracy: {accuracy:.2%}")
        print(f"   F1-Score (macro): {f1_macro:.4f}")
        
        print(f"\n2. PER-CLASS PERFORMANCE:")
        for i, type_name in enumerate(['Type L (Low)', 'Type M (Medium)', 'Type H (High)']):
            class_acc = cm_normalized[i, i]
            total_samples = cm[i].sum()
            correct = cm[i, i]
            print(f"   {type_name}:")
            print(f"     Accuracy: {class_acc:.2%} ({correct}/{total_samples})")
        
        print(f"\n3. MOST IMPORTANT FEATURES:")
        top_features = self.results['feature_importance'].head(5)
        for idx, row in top_features.iterrows():
            print(f"   {row['feature']}: {row['importance']:.4f}")
        
        print(f"\n4. CONCLUSION:")
        if accuracy >= 0.90:
            print("   ✓ EXCELLENT: The model can accurately classify machine type")
            print("     based on operating conditions and failure patterns.")
        elif accuracy >= 0.75:
            print("   ✓ GOOD: The model shows strong ability to classify type")
            print("     from sensor data and failures.")
        elif accuracy >= 0.60:
            print("   ⚠ MODERATE: The model has some predictive ability but")
            print("     may struggle with certain type distinctions.")
        else:
            print("   ✗ POOR: Type may not be strongly determined by")
            print("     the available sensor and failure features.")
        
        # Check if specific types are problematic
        worst_class = np.argmin(cm_normalized.diagonal())
        worst_acc = cm_normalized[worst_class, worst_class]
        type_names = ['L', 'M', 'H']
        
        if worst_acc < 0.7:
            print(f"\n   Note: Type {type_names[worst_class]} is most difficult to classify")
            print(f"   (only {worst_acc:.2%} accuracy)")
        
        print(f"\n5. PRACTICAL IMPLICATIONS:")
        print("   This model allows manufacturers to:")
        print("   • Identify machine quality type from operational data")
        print("   • Understand which conditions indicate different quality levels")
        print("   • Reverse-engineer quality classification from sensor readings")
        
        return self
    
    def save_model(self, model_path='outputs/models/nn_type_classifier.h5'):
        """Save trained model"""
        import os
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        self.model.save(model_path)
        print(f"\n✓ Model saved to: {model_path}")
        return self


def main():
    """Main execution function"""
    # Configuration
    DATA_PATH = 'data/ai4i2020.csv'
    
    SENSOR_FEATURES = [
        'Air temperature [K]',
        'Process temperature [K]',
        'Rotational speed [rpm]',
        'Torque [Nm]',
        'Tool wear [min]'
    ]
    
    FAILURE_FEATURES = [
        'Machine failure',
        'TWF',  # Tool Wear Failure
        'HDF',  # Heat Dissipation Failure
        'PWF',  # Power Failure
        'OSF',  # Overstrain Failure
        'RNF'   # Random Failure
    ]
    
    print("="*60)
    print("NEURAL NETWORK TYPE CLASSIFIER")
    print("Predicting Type (L/M/H) from conditions and failures")
    print("="*60)
    
    # Initialize classifier
    classifier = TypeClassifierNN(
        data_path=DATA_PATH,
        target='Type',
        sensor_features=SENSOR_FEATURES,
        failure_features=FAILURE_FEATURES,
        random_state=42
    )
    
    # Run complete pipeline
    (classifier
     .load_and_prepare_data()
     .prepare_train_test_split(test_size=0.2, val_size=0.2)
     .build_model(hidden_layers=[256, 128, 64], dropout_rate=0.4, learning_rate=0.001)
     .train_model(epochs=150, batch_size=64, patience=20)
     .evaluate_model()
     .analyze_feature_importance()
     .visualize_results()
     .generate_summary()
     .save_model())
    
    return classifier


if __name__ == "__main__":
    classifier = main()
    print("\n" + "="*60)
    print("Analysis complete!")
    print("="*60)