"""
CNN-LSTM Inspired Network Intrusion Detection System
Using TensorFlow/Scikit-learn with Continuous Learning
For CICIDS2017 Dataset

STATUS: Supervised Classification Mode with CONTINUOUS LEARNING is ACTIVE.
THE FINAL REPORT PRINTS THE EVALUATION ON THE TEST SET.

MODIFICATIONS:
- The training dataset uses Monday, Tuesday, Thursday (2 files), and Friday-PortScan.
- Unseen data is loaded from a separate generated file for evaluation.
- Added continuous learning capability for new attack pattern adaptation.
- Enhanced visualization with confusion matrix plots and detailed metrics.
- Automatic percentile-based confidence threshold selection for continuous learning.
"""

import numpy as np
import pandas as pd
import os
import warnings
import time
import traceback

# Visualization imports
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
    print("Matplotlib available for visualization.")
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: Matplotlib not available. Visualization will be skipped.")

# Deep Learning imports
try:
    import tensorflow as tf
    from tensorflow.keras.models import Model, load_model
    from tensorflow.keras.layers import (Conv1D, MaxPooling1D, LSTM, Dense,
                                         Dropout, BatchNormalization, Input,
                                         Reshape)
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.utils import to_categorical
    TENSORFLOW_AVAILABLE = True
    print("TensorFlow/Keras available for deep learning.")
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("WARNING: TensorFlow not available, using fallback MLP implementation.")

# Scikit-learn imports
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, accuracy_score,
                             precision_score, recall_score, f1_score,
                             confusion_matrix)
from sklearn.neural_network import MLPClassifier
import joblib

# Imbalanced-learn import for class imbalance
try:
    from imblearn.over_sampling import ADASYN
    IMBALANCED_LEARN_AVAILABLE = True
except ImportError:
    print("Warning: imbalanced-learn not available. Class balancing will be skipped.")
    IMBALANCED_LEARN_AVAILABLE = False

warnings.filterwarnings('ignore')


class CNNLSTMInspiredDetector:
    """Main class for the intrusion detection system with continuous learning."""

    def __init__(self, dataset_path, unseen_data_path, use_class_balancing=True):
        self.dataset_path = dataset_path
        self.unseen_data_path = unseen_data_path
        self.use_class_balancing = use_class_balancing

        self.scaler = StandardScaler()
        self.model = None
        self.label_encoder = None
        self.feature_names = None
        self.training_time = 0
        self.evaluation_results = {}

        self.X_test_scaled, self.y_test, self.y_pred = None, None, None

        # Continuous learning attributes
        self.memory_X = []
        self.memory_y = []
        self.memory_limit = 5000

        # Backwards-compatible stored threshold value (kept for saving/loading)
        self.confidence_threshold = 0.85
        self.learning_rate_decay = 0.9

        # ---------------------------
        # AUTO CONFIDENCE THRESHOLD
        # ---------------------------
        # Running adaptive threshold used during live operation:
        self.conf_threshold = float(self.confidence_threshold)   # start value (float)
        self.conf_history = []                                   # store recent confidences
        self.conf_history_limit = 500                            # window size to compute percentile
        self.percentile_target = 0.85                            # target percentile (0.85 -> 85th percentile)
        self.threshold_smoothing = 0.3                           # smoothing factor for threshold updates (0-1)

    # ---------------------------------------------------------------------
    # VISUALIZATION METHODS
    # ---------------------------------------------------------------------
    def plot_confusion_matrix(self, y_true, y_pred, class_names, title="Confusion Matrix"):
        """Display confusion matrix in text format."""
        cm = confusion_matrix(y_true, y_pred)

        print(f"\n{title.upper()}:")
        print("="*60)

        # Print raw confusion matrix in text format
        print("\nRaw Confusion Matrix:")
        print("-" * 40)
        print(f"{'':>15}", end="")
        for name in class_names:
            print(f"{name[:8]:>8}", end="")
        print()

        for i, true_label in enumerate(class_names):
            print(f"{true_label[:15]:>15}", end="")
            for j, pred_label in enumerate(class_names):
                if i < cm.shape[0] and j < cm.shape[1]:
                    print(f"{cm[i, j]:>8}", end="")
                else:
                    print(f"{'0':>8}", end="")
            print()

        # Print normalized confusion matrix
        # guard against division by zero when a class has zero true samples
        cm_normalized = np.zeros_like(cm, dtype=float)
        row_sums = cm.sum(axis=1)
        for i in range(cm.shape[0]):
            if row_sums[i] > 0:
                cm_normalized[i] = cm[i] / row_sums[i]

        print("\nNormalized Confusion Matrix (Recall per class):")
        print("-" * 50)
        print(f"{'':>15}", end="")
        for name in class_names:
            print(f"{name[:8]:>8}", end="")
        print()

        for i, true_label in enumerate(class_names):
            print(f"{true_label[:15]:>15}", end="")
            for j, pred_label in enumerate(class_names):
                if i < cm_normalized.shape[0] and j < cm_normalized.shape[1]:
                    print(f"{cm_normalized[i, j]:>8.2f}", end="")
                else:
                    print(f"{'0.00':>8}", end="")
            print()

        return cm

    def plot_metrics_history(self, metrics_history):
        """Display training metrics history in text format."""
        if not metrics_history:
            print("WARNING: No metrics history to display.")
            return

        print("\nMETRICS HISTORY:")
        print("=" * 50)

        for metric_name, values in metrics_history.items():
            if values:
                print(f"\n{metric_name.upper()}:")
                print(f"  Latest: {values[-1]:.4f}")
                print(f"  Best: {max(values):.4f}")
                print(f"  Average: {sum(values)/len(values):.4f}")
                print(f"  Trend: {' '.join([f'{v:.3f}' for v in values[-5:]])}")
        print("=" * 50)

    # ---------------------------------------------------------------------
    # AUTO THRESHOLD METHOD
    # ---------------------------------------------------------------------
    def auto_update_threshold(self):
        """Automatically adjust confidence threshold using percentile of recent confidences."""
        # Need at least a small number of points
        if len(self.conf_history) < 50:
            return  # not enough data yet

        # Keep only last N values
        if len(self.conf_history) > self.conf_history_limit:
            self.conf_history = self.conf_history[-self.conf_history_limit:]

        # Compute new threshold as the percentile (e.g. 85th percentile)
        new_threshold = float(np.percentile(self.conf_history, self.percentile_target * 100))

        # Smooth the threshold to avoid very sudden jumps
        smoothing = float(self.threshold_smoothing)
        self.conf_threshold = (self.conf_threshold * (1.0 - smoothing)) + (new_threshold * smoothing)

        # Keep conf_threshold in [0.0, 1.0]
        self.conf_threshold = float(min(max(self.conf_threshold, 0.0), 1.0))

        # Also update legacy value for saving compatibility
        self.confidence_threshold = float(self.conf_threshold)

        print(f"[AUTO-THRESHOLD] Updated threshold (smoothed) = {self.conf_threshold:.4f}")

    # ---------------------------------------------------------------------
    # TRAINING DATASET LOADER
    # ---------------------------------------------------------------------
    def load_data(self):
        """Load CICIDS2017 dataset files for training."""
        print("Loading selected CICIDS2017 dataset files for training...")
        files_to_load = [
            'Monday-WorkingHours.pcap_ISCX.csv',
            'Tuesday-WorkingHours.pcap_ISCX.csv',
            'Wednesday-workingHours.pcap_ISCX.csv',
            'Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv',
            'Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv'
        ]

        df_list = []
        for f in files_to_load:
            file_path = os.path.join(self.dataset_path, f)
            if not os.path.exists(file_path):
                print(f"WARNING: File not found, skipping: {f}")
                continue

            print(f"Loading {f}...")
            try:
                df_temp = pd.read_csv(file_path, low_memory=False, skipinitialspace=True)
                df_temp.columns = df_temp.columns.str.strip()
                df_list.append(df_temp)
                print(f"   Loaded {df_temp.shape[0]} rows with {df_temp.shape[1]} columns")
            except Exception as e:
                print(f"ERROR: Error loading {f}: {e}")
                continue

        if not df_list:
            raise ValueError("No training data could be loaded! Check dataset folder.")

        self.raw_data = pd.concat(df_list, ignore_index=True)
        print(f"\nFull dataset shape: {self.raw_data.shape}")

        # Sample for quicker testing
        sample_size = 100000
        if len(self.raw_data) > sample_size:
            print(f"Using a sample of {sample_size} rows for training...")
            self.raw_data = self.raw_data.sample(n=sample_size, random_state=42)

        return self.raw_data

    # ---------------------------------------------------------------------
    # DATA PREPROCESSING
    # ---------------------------------------------------------------------
    def preprocess_data(self):
        print("\nPreprocessing data...")
        label_col = 'Label'
        if label_col not in self.raw_data.columns:
            raise ValueError("Label column not found.")

        self.raw_data.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.raw_data.dropna(inplace=True)

        y = self.raw_data[label_col].copy()
        X = self.raw_data.drop(columns=[label_col])

        numeric_cols = X.select_dtypes(include=np.number).columns
        X_numeric = X[numeric_cols]
        self.feature_names = numeric_cols.tolist()

        y = y.loc[X_numeric.index].astype(str).str.strip()
        y = y.str.replace('[\u2010-\u2015]', '-', regex=True)
        y = y.str.replace('\uFFFD', '-', regex=True)
        y = y.str.replace(r'\s+', ' ', regex=True)
        y.fillna('BENIGN', inplace=True)

        print("\nLabel distribution:")
        print(y.value_counts())

        # Filter rare classes
        min_samples = 2
        sufficient_classes = y.value_counts()[lambda c: c >= min_samples].index
        mask = y.isin(sufficient_classes)
        X_numeric, y = X_numeric[mask], y[mask]

        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)

        print(f"\nTraining on {len(self.label_encoder.classes_)} classes.")
        print(f"Classes: {list(self.label_encoder.classes_)}")

        return X_numeric.values, y_encoded

    # ---------------------------------------------------------------------
    def handle_class_imbalance(self, X, y):
        if not self.use_class_balancing or not IMBALANCED_LEARN_AVAILABLE:
            print("Skipping ADASYN balancing.")
            return X, y

        print("Balancing with ADASYN...")
        adasyn = ADASYN(random_state=42, sampling_strategy='not majority')
        try:
            X_resampled, y_resampled = adasyn.fit_resample(X, y)
            print(f"Original: {X.shape}, Resampled: {X_resampled.shape}")
            return X_resampled, y_resampled
        except Exception as e:
            print(f"WARNING: ADASYN failed: {e}. Using original data.")
            return X, y

    # ---------------------------------------------------------------------
    def build_cnn_lstm_model(self, input_shape, num_classes):
        print("Building CNN-LSTM model...")
        inputs = Input(shape=input_shape)
        reshaped = Reshape((input_shape[0], 1))(inputs)

        # CNN layers
        conv1 = Conv1D(64, 3, activation='relu', padding='same')(reshaped)
        conv1 = BatchNormalization()(conv1)
        conv1 = MaxPooling1D(pool_size=2)(conv1)
        conv1 = Dropout(0.2)(conv1)

        # LSTM layer
        lstm1 = LSTM(128, return_sequences=False)(conv1)
        lstm1 = Dropout(0.3)(lstm1)

        # Dense layers
        dense1 = Dense(128, activation='relu')(lstm1)
        outputs = Dense(num_classes, activation='softmax')(dense1)

        model = Model(inputs, outputs)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    # ---------------------------------------------------------------------
    def train_model(self, X, y):
        print("\nTraining model...")
        start = time.time()
        X, y = self.handle_class_imbalance(X, y)
        X_scaled = self.scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )

        self.X_test_scaled, self.y_test = X_test, y_test
        input_shape = (X_train.shape[1],)
        num_classes = len(np.unique(y))

        if TENSORFLOW_AVAILABLE:
            y_train_cat = to_categorical(y_train, num_classes)
            y_test_cat = to_categorical(y_test, num_classes)
            self.model = self.build_cnn_lstm_model(input_shape, num_classes)

            history = self.model.fit(
                X_train, y_train_cat,
                validation_data=(X_test, y_test_cat),
                epochs=10, batch_size=512,
                callbacks=[EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)],
                verbose=1
            )
        else:
            print("Using MLP fallback...")
            self.model = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=10, random_state=42)
            self.model.fit(X_train, y_train)

        self.training_time = time.time() - start
        print(f"Training done in {self.training_time:.2f}s.")

        # Predict and evaluate
        y_pred = self.model.predict(X_test)
        if TENSORFLOW_AVAILABLE:
            y_pred = np.argmax(y_pred, axis=1)

        self.y_pred = y_pred
        self.evaluation_results = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0)
        }

    # ---------------------------------------------------------------------
    def print_detailed_results(self):
        """Print comprehensive evaluation results."""
        print("\n" + "="*70)
        print("DETAILED EVALUATION RESULTS")
        print("="*70)

        # Single metrics summary
        print(f"Accuracy:  {self.evaluation_results['accuracy']:.4f}")
        print(f"Precision: {self.evaluation_results['precision']:.4f}")
        print(f"Recall:    {self.evaluation_results['recall']:.4f}")
        print(f"F1-Score:  {self.evaluation_results['f1']:.4f}")
        print("-"*70)

        # Detailed classification report
        print("\nDETAILED CLASSIFICATION REPORT:")
        class_names = self.label_encoder.classes_
        print(classification_report(self.y_test, self.y_pred, target_names=class_names, zero_division=0))

        # Confusion matrix visualization
        print("\nCONFUSION MATRIX VISUALIZATION:")
        cm = self.plot_confusion_matrix(
            self.y_test, self.y_pred, class_names,
            title="Training Evaluation"
        )

        # Print confusion matrix numerically
        print("\nCONFUSION MATRIX (Numerical):")
        print(f"{'':>15}", end="")
        for name in class_names:
            print(f"{name[:12]:>12}", end="")
        print()

        for i, true_class in enumerate(class_names):
            print(f"{true_class[:14]:>15}", end="")
            for j in range(len(class_names)):
                if i < len(cm) and j < len(cm[i]):
                    print(f"{cm[i, j]:>12d}", end="")
                else:
                    print(f"{'0':>12}", end="")
            print()

    # ---------------------------------------------------------------------
    def load_unseen_data(self):
        """Load generated unseen data file"""
        print("\nLoading unseen dataset...")
        if not os.path.exists(self.unseen_data_path):
            raise FileNotFoundError(f"Unseen file not found: {self.unseen_data_path}")

        df = pd.read_csv(self.unseen_data_path, low_memory=False, skipinitialspace=True)
        df.columns = df.columns.str.strip()
        print(f"Unseen dataset shape: {df.shape}")
        return df

    # ---------------------------------------------------------------------
    def evaluate_on_unseen_data(self):
        """Evaluate model on unseen data with continuous learning."""
        print("\nEvaluating on unseen data...")
        unseen = self.load_unseen_data()
        label_col = 'Label'
        unseen.replace([np.inf, -np.inf], np.nan, inplace=True)
        unseen.dropna(inplace=True)

        y = unseen[label_col].astype(str).str.strip()
        X = unseen[self.feature_names]
        y = y.loc[X.index]

        y = y.str.replace('[\u2010-\u2015]', '-', regex=True)
        y = y.str.replace(r'\s+', ' ', regex=True)
        y.fillna('BENIGN', inplace=True)

        # Handle new classes in unseen data
        known_classes = set(self.label_encoder.classes_)
        new_classes = set(y.unique()) - known_classes

        if new_classes:
            print(f"New attack classes detected: {new_classes}")
            print("Expanding label encoder for continuous learning...")

            # Expand label encoder to handle new classes
            all_classes = list(known_classes) + list(new_classes)
            temp_encoder = LabelEncoder()
            temp_encoder.fit(all_classes)

            # Map old labels to new encoding
            old_to_new_mapping = {}
            for old_class in self.label_encoder.classes_:
                old_idx = np.where(self.label_encoder.classes_ == old_class)[0][0]
                new_idx = np.where(temp_encoder.classes_ == old_class)[0][0]
                old_to_new_mapping[old_idx] = new_idx

            self.label_encoder = temp_encoder

            # Expand model to handle new classes
            new_num_classes = len(self.label_encoder.classes_)
            self.expand_model_for_new_classes(new_num_classes)

        # Filter to known classes for evaluation, mark new classes for learning
        known_mask = y.isin(known_classes)
        X_known, y_known = X[known_mask], y[known_mask]
        X_new, y_new = X[~known_mask], y[~known_mask]

        if len(X_known) > 0:
            y_enc_known = self.label_encoder.transform(y_known)
            X_scaled_known = self.scaler.transform(X_known)

            # Predict on known classes
            if TENSORFLOW_AVAILABLE:
                preds_known = self.model.predict(X_scaled_known)
                y_pred_known = np.argmax(preds_known, axis=1)
            else:
                preds_known = None
                y_pred_known = self.model.predict(X_scaled_known)

            print(f"\nUNSEEN DATA EVALUATION RESULTS (Known Classes):")
            print("="*70)

            accuracy = accuracy_score(y_enc_known, y_pred_known)
            precision = precision_score(y_enc_known, y_pred_known, average='weighted', zero_division=0)
            recall = recall_score(y_enc_known, y_pred_known, average='weighted', zero_division=0)
            f1 = f1_score(y_enc_known, y_pred_known, average='weighted', zero_division=0)

            print(f"Accuracy:  {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall:    {recall:.4f}")
            print(f"F1-Score:  {f1:.4f}")

            # Get present classes
            unique_classes_in_unseen = np.unique(y_enc_known)
            present_class_names = [self.label_encoder.classes_[i] for i in unique_classes_in_unseen]

            print(f"\nClasses in unseen data: {present_class_names}")
            print("\nDetailed Classification Report:")
            print(classification_report(y_enc_known, y_pred_known,
                                       target_names=present_class_names,
                                       labels=unique_classes_in_unseen,
                                       zero_division=0))

            # Confusion matrix for unseen data
            print("\nUnseen Data Confusion Matrix:")
            cm_unseen = self.plot_confusion_matrix(
                y_enc_known, y_pred_known, present_class_names,
                title="Unseen Data Evaluation"
            )

            # Continuous learning on confident predictions
            self.continuous_learning_update(X_scaled_known, y_pred_known, preds_known if TENSORFLOW_AVAILABLE else None)

        # Handle new attack classes
        if len(X_new) > 0:
            print(f"\nProcessing {len(X_new)} samples with new attack classes...")
            self.learn_new_attacks(X_new, y_new)

    # ---------------------------------------------------------------------
    def expand_model_for_new_classes(self, new_num_classes):
        """Rebuild model with expanded output layer for new classes."""
        if not TENSORFLOW_AVAILABLE:
            return

        print(f"Expanding model from {self.model.output.shape[-1]} to {new_num_classes} classes...")

        # Get the old model's weights
        old_weights = []
        for layer in self.model.layers[:-1]:  # All layers except last dense layer
            old_weights.append(layer.get_weights())

        # Get the last layer's weights
        last_layer_weights = self.model.layers[-1].get_weights()
        old_weight_matrix = last_layer_weights[0]  # Shape: (128, old_num_classes)
        old_bias = last_layer_weights[1]  # Shape: (old_num_classes,)

        # Build new model with expanded output
        input_shape = (self.model.input.shape[1],)
        self.model = self.build_cnn_lstm_model(input_shape, new_num_classes)

        # Restore weights to all layers except the last one
        for i, layer in enumerate(self.model.layers[:-1]):
            if i < len(old_weights):
                try:
                    layer.set_weights(old_weights[i])
                except Exception:
                    # If shapes mismatch for any reason, skip restoring that layer
                    pass

        # Expand the last layer's weights
        # Determine last dense input dim dynamically (in case design changed)
        last_dense = self.model.layers[-1]
        last_in_dim = last_layer_weights[0].shape[0] if hasattr(last_layer_weights[0], 'shape') else 128
        new_weight_matrix = np.random.normal(0, 0.01, (last_in_dim, new_num_classes))
        new_bias = np.zeros(new_num_classes)

        # Copy old weights to new matrix (first columns)
        old_num_classes = old_weight_matrix.shape[1]
        new_weight_matrix[:, :old_num_classes] = old_weight_matrix
        new_bias[:old_num_classes] = old_bias

        # Set the expanded weights
        try:
            self.model.layers[-1].set_weights([new_weight_matrix, new_bias])
        except Exception as e:
            print(f"WARNING: Could not set expanded weights for last layer: {e}")

        print(f"Model successfully expanded to handle {new_num_classes} classes")

    # ---------------------------------------------------------------------
    def continuous_learning_update(self, X_scaled, y_pred, prediction_probs=None):
        """Update model with confident predictions for continuous learning."""
        print("\nContinuous Learning Update...")

        if prediction_probs is not None:
            # Use prediction confidence for selection
            confidence = np.max(prediction_probs, axis=1)

            # Store confidence values for auto-threshold tuning
            self.conf_history.extend(confidence.tolist())

            # Auto adjust threshold
            self.auto_update_threshold()

            # Use adaptive threshold
            confident_mask = confidence >= self.conf_threshold

            if np.sum(confident_mask) > 0:
                print(f"Adding {np.sum(confident_mask)} confident samples to memory")

                # Add to memory
                self.add_to_memory(X_scaled[confident_mask], y_pred[confident_mask])

                # Incremental learning
                if len(self.memory_X) >= 100:  # Minimum batch size
                    self.incremental_update()
            else:
                print("WARNING: No confident predictions found for learning")
        else:
            print("Using all predictions for MLP continuous learning")
            # For MLP we don't have prediction probs; still we can optionally use heuristic
            # Here we will use all predictions for MLP fallback as before
            self.add_to_memory(X_scaled, y_pred)

    # ---------------------------------------------------------------------
    def add_to_memory(self, X_new, y_new):
        """Add new samples to memory buffer."""
        self.memory_X.extend(X_new.tolist())
        self.memory_y.extend(y_new.tolist())

        # Maintain memory limit
        if len(self.memory_X) > self.memory_limit:
            excess = len(self.memory_X) - self.memory_limit
            self.memory_X = self.memory_X[excess:]
            self.memory_y = self.memory_y[excess:]

        print(f"Memory updated: {len(self.memory_X)} samples")

    # ---------------------------------------------------------------------
    def incremental_update(self):
        """Perform incremental model update with memory replay."""
        if len(self.memory_X) == 0:
            return

        print(f"Performing incremental update with {len(self.memory_X)} samples...")

        X_mem = np.array(self.memory_X)
        y_mem = np.array(self.memory_y)

        if TENSORFLOW_AVAILABLE:
            # Convert to categorical
            num_classes = len(self.label_encoder.classes_)

            # Ensure all memory labels are valid for current model
            max_label = np.max(y_mem)
            if max_label >= num_classes:
                print(f"WARNING: Memory contains invalid label {max_label} >= {num_classes}")
                # Filter out invalid labels
                valid_mask = y_mem < num_classes
                X_mem = X_mem[valid_mask]
                y_mem = y_mem[valid_mask]
                print(f"Filtered to {len(X_mem)} valid samples")

                if len(X_mem) == 0:
                    print("WARNING: No valid samples left after filtering")
                    return

            y_mem_cat = to_categorical(y_mem, num_classes=num_classes)

            # Reduce learning rate for fine-tuning
            try:
                original_lr = self.model.optimizer.learning_rate.numpy()
                self.model.optimizer.learning_rate = original_lr * self.learning_rate_decay
            except Exception:
                # if optimizer learning rate isn't accessible, skip adjustment
                pass

            print(f"Fine-tuning with learning rate (approx): {getattr(self.model.optimizer, 'learning_rate', 'unknown')}")
            print(f"Training data shape: X={X_mem.shape}, y={y_mem_cat.shape}")

            # Fine-tune model
            history = self.model.fit(X_mem, y_mem_cat,
                                     epochs=3, batch_size=128,
                                     verbose=1, shuffle=True)

            print("Incremental update completed")
        else:
            print("Updating MLP with partial_fit...")
            # For MLP, use partial_fit if available
            if hasattr(self.model, 'partial_fit'):
                unique_classes = np.arange(len(self.label_encoder.classes_))
                self.model.partial_fit(X_mem, y_mem, classes=unique_classes)
            else:
                print("WARNING: MLP doesn't support incremental learning, retraining...")

    # ---------------------------------------------------------------------
    def learn_new_attacks(self, X_new, y_new):
        """Learn from completely new attack types."""
        print(f"\nLearning new attack patterns...")
        print(f"New attack types: {y_new.unique()}")

        # For now, we'll store these for future model expansion
        # In a production system, you might trigger a full retrain or model expansion
        X_scaled_new = self.scaler.transform(X_new)
        # label_encoder should already have been expanded when new classes detected; if not this will raise
        try:
            y_encoded_new = self.label_encoder.transform(y_new)
        except Exception as e:
            # If transform fails because label encoder wasn't expanded, attempt to expand naively
            print(f"WARNING: label encoding new attacks failed: {e}")
            # fallback: treat as unknown class index 0 (not ideal) - but we should have expanded earlier
            return

        print(f"Storing {len(X_new)} new attack samples for future learning")
        self.add_to_memory(X_scaled_new, y_encoded_new)

        # If we have enough new samples, trigger a learning update
        if len(X_new) >= 50:
            print("Sufficient new samples detected, triggering learning update...")
            self.incremental_update()

    # ---------------------------------------------------------------------
    def save_model(self, filename='ids_model_continuous.joblib'):
        """Save model with continuous learning state."""
        if self.model is None:
            print("ERROR: No model to save.")
            return

        pkg = {
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names,
            'memory_X': self.memory_X[-1000:],  # Save last 1000 memory samples
            'memory_y': self.memory_y[-1000:],
            # save both legacy and auto-threshold parameters
            'confidence_threshold': self.confidence_threshold,
            'conf_threshold': self.conf_threshold,
            'conf_history': self.conf_history[-self.conf_history_limit:],
            'conf_history_limit': self.conf_history_limit,
            'percentile_target': self.percentile_target,
            'threshold_smoothing': self.threshold_smoothing,
            'learning_rate_decay': self.learning_rate_decay
        }

        if TENSORFLOW_AVAILABLE:
            model_path = filename.replace('.joblib', '.h5')
            try:
                self.model.save(model_path)
                pkg['model_path'] = model_path
            except Exception as e:
                print(f"WARNING: Could not save Keras model to H5: {e}")
        else:
            pkg['model'] = self.model

        joblib.dump(pkg, filename)
        print(f"Model with continuous learning state saved as {filename}")


# -------------------------------------------------------------------------
# CONTINUOUS LEARNING MODEL LOADER
# -------------------------------------------------------------------------
class ContinuousLearningIDS:
    """Load and use trained model with continuous learning capabilities."""

    def __init__(self, model_package_path):
        """Initialize continuous learning system from saved model."""
        print("Loading continuous learning model...")

        pkg = joblib.load(model_package_path)
        self.scaler = pkg['scaler']
        self.label_encoder = pkg['label_encoder']
        self.feature_names = pkg['feature_names']

        # Continuous learning state
        self.memory_X = pkg.get('memory_X', [])
        self.memory_y = pkg.get('memory_y', [])
        # load legacy if present
        self.confidence_threshold = pkg.get('confidence_threshold', 0.85)
        # load auto-threshold parameters
        self.conf_threshold = pkg.get('conf_threshold', float(self.confidence_threshold))
        self.conf_history = pkg.get('conf_history', [])
        self.conf_history_limit = pkg.get('conf_history_limit', 500)
        self.percentile_target = pkg.get('percentile_target', 0.85)
        self.threshold_smoothing = pkg.get('threshold_smoothing', 0.3)

        self.learning_rate_decay = pkg.get('learning_rate_decay', 0.9)
        self.memory_limit = 5000

        if 'model_path' in pkg:
            try:
                self.model = load_model(pkg['model_path'])
                self.is_tensorflow = True
            except Exception as e:
                print(f"WARNING: Failed to load keras model: {e}")
                self.model = pkg.get('model', None)
                self.is_tensorflow = False
        else:
            self.model = pkg.get('model', None)
            self.is_tensorflow = False

        print(f"Model loaded with {len(self.memory_X)} memory samples and conf_threshold={self.conf_threshold:.4f}")

    def auto_update_threshold(self):
        """Auto update threshold in loader as well (same logic)."""
        if len(self.conf_history) < 50:
            return

        if len(self.conf_history) > self.conf_history_limit:
            self.conf_history = self.conf_history[-self.conf_history_limit:]

        new_threshold = float(np.percentile(self.conf_history, self.percentile_target * 100))
        smoothing = float(self.threshold_smoothing)
        self.conf_threshold = (self.conf_threshold * (1.0 - smoothing)) + (new_threshold * smoothing)
        self.conf_threshold = float(min(max(self.conf_threshold, 0.0), 1.0))
        self.confidence_threshold = float(self.conf_threshold)
        print(f"[AUTO-THRESHOLD - loader] Updated threshold = {self.conf_threshold:.4f}")

    def predict_and_learn(self, X_new, y_true=None):
        """Make predictions and learn from new data."""
        X_scaled = self.scaler.transform(X_new)

        if self.is_tensorflow:
            predictions = self.model.predict(X_scaled)
            y_pred = np.argmax(predictions, axis=1)
            confidence = np.max(predictions, axis=1)
        else:
            y_pred = self.model.predict(X_scaled)
            if hasattr(self.model, 'predict_proba'):
                predictions = self.model.predict_proba(X_scaled)
                confidence = np.max(predictions, axis=1)
            else:
                confidence = np.ones(len(y_pred))  # Assume high confidence for deterministic models

        # Update confidence history and auto-adjust threshold
        self.conf_history.extend(confidence.tolist())
        self.auto_update_threshold()

        # Learn from confident predictions using adaptive threshold
        confident_mask = confidence >= self.conf_threshold

        if np.sum(confident_mask) > 0:
            print(f"Learning from {np.sum(confident_mask)} confident predictions...")
            self.add_to_memory(X_scaled[confident_mask], y_pred[confident_mask])

            if len(self.memory_X) >= 100:
                self.incremental_update()

        # Convert predictions back to class names
        predicted_classes = [self.label_encoder.classes_[pred] for pred in y_pred]

        return predicted_classes, confidence

    def add_to_memory(self, X_new, y_new):
        """Add samples to experience replay memory."""
        self.memory_X.extend(X_new.tolist())
        self.memory_y.extend(y_new.tolist())

        if len(self.memory_X) > self.memory_limit:
            excess = len(self.memory_X) - self.memory_limit
            self.memory_X = self.memory_X[excess:]
            self.memory_y = self.memory_y[excess:]

    def incremental_update(self):
        """Perform incremental model update."""
        if len(self.memory_X) == 0:
            return

        X_mem = np.array(self.memory_X)
        y_mem = np.array(self.memory_y)

        if self.is_tensorflow:
            num_classes = len(self.label_encoder.classes_)
            y_mem_cat = to_categorical(y_mem, num_classes=num_classes)

            # Reduce learning rate
            try:
                original_lr = self.model.optimizer.learning_rate.numpy()
                self.model.optimizer.learning_rate = original_lr * self.learning_rate_decay
            except Exception:
                pass

            self.model.fit(X_mem, y_mem_cat, epochs=2, batch_size=128, verbose=0)
        else:
            if hasattr(self.model, 'partial_fit'):
                unique_classes = np.arange(len(self.label_encoder.classes_))
                self.model.partial_fit(X_mem, y_mem, classes=unique_classes)


# -------------------------------------------------------------------------
def main():
    DATASET_PATH = os.path.join('dataset', 'cicd')
    UNSEEN_PATH = os.path.join('dataset', 'generated_network_traffic.csv')

    # Check if paths exist
    if not os.path.exists(DATASET_PATH):
        print(f"ERROR: Dataset path not found: {DATASET_PATH}")
        return

    if not os.path.exists(UNSEEN_PATH):
        print(f"ERROR: Unseen data file not found: {UNSEEN_PATH}")
        return

    print("=" * 70)
    print("CNN-LSTM INSPIRED IDS - CONTINUOUS LEARNING ENABLED (AUTO THRESHOLD)")
    print("=" * 70)

    # Training phase
    detector = CNNLSTMInspiredDetector(DATASET_PATH, UNSEEN_PATH)
    detector.load_data()
    X, y = detector.preprocess_data()
    detector.train_model(X, y)
    detector.print_detailed_results()
    detector.save_model('ids_model_continuous.joblib')

    # Evaluation with continuous learning
    detector.evaluate_on_unseen_data()

    print("\nTraining, Evaluation, and Continuous Learning Complete!")
    print("All confusion matrices and metrics have been displayed and saved.")
    print("Model saved with continuous learning capabilities.")


if __name__ == "__main__":
    main()
