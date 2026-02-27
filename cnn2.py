"""
CNN-LSTM IDS with Continuous Learning (Auto-Save Enabled)
Implements pseudo-labeling, experience replay, and automatic model saving after each incremental update.
Memory-efficient version using float32 and dataset sampling.
"""

import os
import time
import warnings
import traceback
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Conv1D, MaxPooling1D, LSTM, Dense,
    Dropout, BatchNormalization, Input, Reshape
)
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, classification_report,
    confusion_matrix
)

warnings.filterwarnings('ignore')


# =====================================================================
# CNN-LSTM MAIN IDS MODEL
# =====================================================================
class CNNLSTMInspiredDetector:
    def __init__(self, dataset_path, unseen_data_path, use_class_balancing=False):
        self.dataset_path = dataset_path
        self.unseen_data_path = unseen_data_path
        self.use_class_balancing = use_class_balancing
        self.scaler = StandardScaler()
        self.label_encoder = None
        self.model = None
        self.feature_names = None
        self.evaluation_results = {}

    # -----------------------------------------------------------
    def load_data(self):
        print("Loading CICIDS2017 training data...")
        files = [
            'Monday-WorkingHours.pcap_ISCX.csv',
            'Tuesday-WorkingHours.pcap_ISCX.csv',
            'Wednesday-workingHours.pcap_ISCX.csv',
            'Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv',
            'Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv'
        ]
        df_list = []
        for f in files:
            path = os.path.join(self.dataset_path, f)
            if not os.path.exists(path):
                print(f"⚠️ Missing file: {f}")
                continue
            print(f"✅ Loading {f} ...")
            df = pd.read_csv(path, low_memory=False)
            df.columns = df.columns.str.strip()
            df_list.append(df)
        if not df_list:
            raise FileNotFoundError("No training CSV files found.")
        self.raw_data = pd.concat(df_list, ignore_index=True)
        print(f"Full dataset shape: {self.raw_data.shape}")
        return self.raw_data

    # -----------------------------------------------------------
    def preprocess_data(self):
        print("\nPreprocessing...")
        label_col = 'Label'
        self.raw_data.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.raw_data.dropna(inplace=True)

        y = self.raw_data[label_col].astype(str).str.strip()
        X = self.raw_data.drop(columns=[label_col])

        numeric_cols = X.select_dtypes(include=np.number).columns
        X = X[numeric_cols].astype(np.float32)  # ✅ save memory
        self.feature_names = numeric_cols.tolist()

        # Optional sampling
        if len(X) > 500000:
            sample_idx = np.random.choice(len(X), 500000, replace=False)
            X = X.iloc[sample_idx]
            y = y.iloc[sample_idx]
            print(f"📉 Using subset of {len(X)} samples for training to save memory.")

        y = y.fillna('BENIGN')
        self.label_encoder = LabelEncoder()
        y_enc = self.label_encoder.fit_transform(y)

        print(f"Classes: {list(self.label_encoder.classes_)}")
        return X.values, y_enc

    # -----------------------------------------------------------
    def build_cnn_lstm_model(self, input_shape, num_classes):
        print("Building CNN-LSTM model...")
        inputs = Input(shape=input_shape)
        reshaped = Reshape((input_shape[0], 1))(inputs)
        conv1 = Conv1D(64, 3, activation='relu', padding='same')(reshaped)
        conv1 = BatchNormalization()(conv1)
        conv1 = MaxPooling1D(pool_size=2)(conv1)
        conv1 = Dropout(0.2)(conv1)
        lstm1 = LSTM(128, return_sequences=False)(conv1)
        lstm1 = Dropout(0.3)(lstm1)
        dense1 = Dense(128, activation='relu')(lstm1)
        outputs = Dense(num_classes, activation='softmax')(dense1)
        model = Model(inputs, outputs)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    # -----------------------------------------------------------
    def handle_class_imbalance(self, X, y):
        """Handle class imbalance using ADASYN or undersampling"""
        print("\n🔄 Handling class imbalance with ADASYN...")
        
        # Check original class distribution
        unique_classes, class_counts = np.unique(y, return_counts=True)
        print("Original class distribution:")
        for cls, count in zip(unique_classes, class_counts):
            print(f"   Class {cls}: {count} samples")
        
        # If use_class_balancing is False, return original data
        if not self.use_class_balancing:
            print("⏭️ Class balancing disabled. Using original data.")
            return X, y
            
        try:
            from imblearn.over_sampling import ADASYN
            from imblearn.under_sampling import RandomUnderSampler
            from imblearn.pipeline import Pipeline as ImblearnPipeline
            
            # Use ADASYN for adaptive oversampling of minority classes
            # ADASYN focuses on hard-to-learn minority class examples
            # and RandomUnderSampler to prevent excessive data generation
            oversample = ADASYN(sampling_strategy='auto', random_state=42, n_neighbors=5)
            undersample = RandomUnderSampler(sampling_strategy='auto', random_state=42)
            
            # Create pipeline: first oversample with ADASYN, then undersample
            pipeline = ImblearnPipeline([
                ('adasyn_oversample', oversample),
                ('undersample', undersample)
            ])
            
            X_resampled, y_resampled = pipeline.fit_resample(X, y)
            
            # Check new class distribution
            unique_classes_new, class_counts_new = np.unique(y_resampled, return_counts=True)
            print("Balanced class distribution:")
            for cls, count in zip(unique_classes_new, class_counts_new):
                print(f"   Class {cls}: {count} samples")
                
            print(f"✅ Data resampled using ADASYN from {X.shape[0]} to {X_resampled.shape[0]} samples")
            print("📊 ADASYN focuses on hard-to-learn minority class examples for better balance")
            return X_resampled, y_resampled
            
        except ImportError:
            print("⚠️ imbalanced-learn not available. Skipping class balancing.")
            return X, y
        except Exception as e:
            print(f"⚠️ Error during ADASYN class balancing: {e}. Using original data.")
            return X, y

    # -----------------------------------------------------------
    def train_model(self, X, y):
        print("\nTraining model...")
        start = time.time()
        
        # Handle class imbalance first
        X, y = self.handle_class_imbalance(X, y)
        X_scaled = self.scaler.fit_transform(X)

        # Check class distribution before splitting
        unique_classes, class_counts = np.unique(y, return_counts=True)
        print(f"📊 Class distribution after balancing:")
        for cls, count in zip(unique_classes, class_counts):
            class_name = self.label_encoder.classes_[cls] if hasattr(self, 'label_encoder') else f"Class_{cls}"
            print(f"   {class_name}: {count} samples")
        
        # Check if any class has less than 2 samples
        min_samples = min(class_counts)
        if min_samples < 2:
            print(f"⚠️ Warning: Some classes have less than 2 samples. Removing classes with < 2 samples.")
            # Remove classes with insufficient samples
            valid_classes = unique_classes[class_counts >= 2]
            valid_mask = np.isin(y, valid_classes)
            X_scaled = X_scaled[valid_mask]
            y = y[valid_mask]
            
            # Update class counts
            unique_classes, class_counts = np.unique(y, return_counts=True)
            print(f"📊 Updated class distribution:")
            for cls, count in zip(unique_classes, class_counts):
                class_name = self.label_encoder.classes_[cls] if hasattr(self, 'label_encoder') else f"Class_{cls}"
                print(f"   {class_name}: {count} samples")

        # Use stratification only if all classes have at least 2 samples
        try:
            if min(np.unique(y, return_counts=True)[1]) >= 2:
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y, test_size=0.2, random_state=42, stratify=y
                )
                print("✅ Used stratified split")
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y, test_size=0.2, random_state=42
                )
                print("⚠️ Used random split (no stratification)")
        except ValueError as e:
            print(f"⚠️ Stratification failed: {e}")
            print("📝 Using random split instead...")
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )

        input_shape = (X_train.shape[1],)
        num_classes = len(np.unique(y))
        y_train_cat = to_categorical(y_train, num_classes)
        y_test_cat = to_categorical(y_test, num_classes)

        print(f"🏗️ Building model: {input_shape} → {num_classes} classes")
        self.model = self.build_cnn_lstm_model(input_shape, num_classes)
        
        print(f"🚀 Starting training...")
        history = self.model.fit(
            X_train, y_train_cat,
            validation_data=(X_test, y_test_cat),
            epochs=8, batch_size=512,
            callbacks=[EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)],
            verbose=1
        )

        # Evaluate on test set
        y_pred = np.argmax(self.model.predict(X_test), axis=1)
        test_accuracy = accuracy_score(y_test, y_pred)
        
        self.training_time = time.time() - start
        self.evaluation_results = {
            'accuracy': test_accuracy,
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0)
        }

        print("\nTraining complete.")
        print("="*50)
        print("📊 SINGLE METRICS SUMMARY")
        print("="*50)
        print(f"🎯 Accuracy:  {self.evaluation_results['accuracy']:.4f}")
        print(f"🔍 Precision: {self.evaluation_results['precision']:.4f}")
        print(f"🎪 Recall:    {self.evaluation_results['recall']:.4f}")
        print(f"⚡ F1-Score:  {self.evaluation_results['f1']:.4f}")
        print("="*50)

    # -----------------------------------------------------------
    def save_model(self, filename='ids_model_package.joblib'):
        print("Saving model package...")
        pkg = {'scaler': self.scaler, 'label_encoder': self.label_encoder, 'feature_names': self.feature_names}
        model_path = filename.replace('.joblib', '.h5')
        self.model.save(model_path)
        pkg['model_path'] = model_path
        joblib.dump(pkg, filename)
        print(f"✅ Model saved as {filename} and {model_path}")

    # -----------------------------------------------------------
    def load_unseen_data(self):
        print("Loading unseen dataset...")
        df = pd.read_csv(self.unseen_data_path, low_memory=False)
        df.columns = df.columns.str.strip()
        return df

    # -----------------------------------------------------------
    def evaluate_on_unseen_data(self):
        """Evaluate model on completely unseen data with comprehensive metrics."""
        print("\n" + "="*70)
        print("📊 EVALUATING ON UNSEEN DATA")
        print("="*70)
        
        df = self.load_unseen_data()
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)

        y = df['Label'].astype(str).str.strip()
        X = df[self.feature_names].astype(np.float32)
        
        # Filter to only include classes seen during training
        mask = y.isin(self.label_encoder.classes_)
        X, y = X[mask], y[mask]
        y_enc = self.label_encoder.transform(y)

        print(f"📋 Unseen data: {len(X)} samples")
        print(f"🏷️ Classes in unseen data: {sorted(y.unique())}")
        print(f"🏷️ Classes in training: {sorted(self.label_encoder.classes_)}")

        X_scaled = self.scaler.transform(X)
        preds = self.model.predict(X_scaled)
        y_pred = np.argmax(preds, axis=1)

        # Calculate comprehensive metrics
        accuracy = accuracy_score(y_enc, y_pred)
        precision_macro = precision_score(y_enc, y_pred, average='macro', zero_division=0)
        precision_micro = precision_score(y_enc, y_pred, average='micro', zero_division=0)
        recall_macro = recall_score(y_enc, y_pred, average='macro', zero_division=0)
        recall_micro = recall_score(y_enc, y_pred, average='micro', zero_division=0)
        f1_macro = f1_score(y_enc, y_pred, average='macro', zero_division=0)
        f1_micro = f1_score(y_enc, y_pred, average='micro', zero_division=0)

        print("\n🎯 OVERALL PERFORMANCE METRICS")
        print("-" * 50)
        print(f"Accuracy:           {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"Precision (Macro):  {precision_macro:.4f}")
        print(f"Precision (Micro):  {precision_micro:.4f}")
        print(f"Recall (Macro):     {recall_macro:.4f}")
        print(f"Recall (Micro):     {recall_micro:.4f}")
        print(f"F1-Score (Macro):   {f1_macro:.4f}")
        print(f"F1-Score (Micro):   {f1_micro:.4f}")

        # Get unique classes that appear in predictions
        unique_classes_in_pred = sorted(set(y_enc) | set(y_pred))
        class_names_filtered = [self.label_encoder.classes_[i] for i in unique_classes_in_pred]

        print(f"\n📊 DETAILED CLASSIFICATION REPORT")
        print("-" * 70)
        try:
            print(classification_report(
                y_enc, y_pred, 
                labels=unique_classes_in_pred,
                target_names=class_names_filtered,
                zero_division=0
            ))
        except Exception as e:
            print(f"⚠️ Classification report error: {e}")
            print("Continuing with confusion matrix...")

        # Confusion Matrix
        print(f"\n🔢 CONFUSION MATRIX")
        print("-" * 50)
        cm = confusion_matrix(y_enc, y_pred, labels=unique_classes_in_pred)
        
        # Print confusion matrix with class names
        print(f"{'':>15}", end="")
        for name in class_names_filtered:
            print(f"{name[:10]:>12}", end="")
        print()
        
        for i, name in enumerate(class_names_filtered):
            print(f"{name[:15]:>15}", end="")
            for j in range(len(class_names_filtered)):
                print(f"{cm[i,j]:>12}", end="")
            print()

        # Per-class accuracy
        print(f"\n📈 PER-CLASS PERFORMANCE")
        print("-" * 50)
        for i, class_name in enumerate(class_names_filtered):
            if i < len(cm):
                class_total = cm[i].sum()
                class_correct = cm[i, i] if i < cm.shape[1] else 0
                class_accuracy = class_correct / class_total if class_total > 0 else 0
                print(f"{class_name:>20}: {class_accuracy:.4f} ({class_correct}/{class_total})")

        # Store results for later use
        self.evaluation_results = {
            'accuracy': accuracy,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'confusion_matrix': cm,
            'class_names': class_names_filtered
        }

        print(f"\n📊 SINGLE EVALUATION METRICS")
        print("="*50)
        print(f"🎯 Accuracy:  {accuracy:.4f}")
        print(f"🔍 Precision: {precision_macro:.4f}")
        print(f"🎪 Recall:    {recall_macro:.4f}")
        print(f"⚡ F1-Score:  {f1_macro:.4f}")
        print("="*50)

        print(f"\n✅ EVALUATION COMPLETED")
        print("="*70)


# =====================================================================
# CONTINUOUS LEARNING WRAPPER (AUTO-SAVE ENABLED)
# =====================================================================
class ContinuousLearningIDS:
    def __init__(self, model_path, scaler_joblib, memory_limit=2000):
        pkg = joblib.load(scaler_joblib)
        self.model = load_model(pkg['model_path'])
        self.scaler = pkg['scaler']
        self.label_encoder = pkg['label_encoder']  # Add label encoder
        self.num_classes = len(self.label_encoder.classes_)  # Get number of classes
        self.memory_X, self.memory_y = [], []
        self.memory_limit = memory_limit
        self.model_path = pkg['model_path']

    def classify_unseen_and_update(self, X_unseen, confidence_thresh=0.92):
        X_scaled = self.scaler.transform(X_unseen)
        preds = self.model.predict(X_scaled)
        conf = np.max(preds, axis=1)
        pseudo_labels = np.argmax(preds, axis=1)

        confident_idx = np.where(conf >= confidence_thresh)[0]
        if len(confident_idx) > 0:
            print(f"Adding {len(confident_idx)} confident samples to memory.")
            self._add_to_memory(X_scaled[confident_idx], pseudo_labels[confident_idx])
            self._fine_tune_with_memory()
            self._auto_save_model()  # ✅ auto-save after update
        else:
            print("No confident samples found.")

    def _add_to_memory(self, X_new, y_new):
        self.memory_X.extend(X_new)
        self.memory_y.extend(y_new)
        if len(self.memory_X) > self.memory_limit:
            self.memory_X = self.memory_X[-self.memory_limit:]
            self.memory_y = self.memory_y[-self.memory_limit:]

    def _fine_tune_with_memory(self, epochs=2, batch_size=128):
        X_mem = np.array(self.memory_X)
        y_mem = tf.keras.utils.to_categorical(np.array(self.memory_y), num_classes=self.num_classes)
        print(f"Fine-tuning on {len(X_mem)} samples...")
        print(f"📊 Memory data shape: X={X_mem.shape}, y={y_mem.shape}")
        self.model.fit(X_mem, y_mem, epochs=epochs, batch_size=batch_size, verbose=1)

    def _auto_save_model(self):
        print("Auto-saving updated model...")
        self.model.save(self.model_path)
        print(f"✅ Model saved at {self.model_path}")


# =====================================================================
# MAIN
# =====================================================================
def main():
    DATASET_PATH = os.path.join('dataset', 'cicd')
    UNSEEN_PATH = os.path.join('dataset', 'generated_network_traffic.csv')

    print("=" * 70)
    print("CNN-LSTM INSPIRED IDS with Continuous Learning")
    print("=" * 70)

    detector = CNNLSTMInspiredDetector(DATASET_PATH, UNSEEN_PATH)
    detector.load_data()
    X, y = detector.preprocess_data()
    detector.train_model(X, y)
    detector.save_model()
    detector.evaluate_on_unseen_data()

    # Continuous learning example (after training)
    print("\n🔁 Starting continuous learning phase...")
    cl = ContinuousLearningIDS('ids_model_package.h5', 'ids_model_package.joblib')
    unseen = detector.load_unseen_data()
    if 'Label' in unseen.columns:
        unseen = unseen.drop(columns=['Label'])
    unseen = unseen[detector.feature_names].astype(np.float32)
    cl.classify_unseen_and_update(unseen.values)

    print("\n✅ Training, Evaluation, and Continuous Learning Completed.")


if __name__ == "__main__":
    main()
