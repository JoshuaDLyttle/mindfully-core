"""
Mindfully AI - Phase 4: Training & Validation Loop
==================================================
File: ml_pipeline/model_trainer.py

Objective:
----------
Ingest the standardized 2-channel prefrontal EEG data (Fp1, Fp2) and train the 
Mindfully deep learning architecture. This script utilizes Transfer Learning to 
freeze the universal salience backbone and fine-tune only the 'substance_urge' head.
"""

import os
import tensorflow as tf
import numpy as np
import mlflow

from data_standardizer import DataStandardizer
from ablation_test import resolve_sample_paths
from craving_classifier import build_mindfully_model, configure_transfer_learning

def extract_raw_windows(raw, window_size=1024, step_size=64):
    """
    Extracts raw 3D time-series windows of shape (n_windows, time_steps, channels).
    """
    # get_data() returns (channels, time_steps). Transpose to (time_steps, channels)
    data = raw.get_data().T 
    n_times, n_channels = data.shape
    
    windows = []
    for start in range(0, n_times - window_size + 1, step_size):
        windows.append(data[start : start + window_size, :])
    
    return np.array(windows)

def create_tf_dataset(X, y, batch_size=8, is_training=True):
    """
    Converts numpy arrays into a highly optimized tf.data.Dataset pipeline.
    """
    y_dict = {
        "substance_urge": y,
        "behavioral_urge": np.zeros_like(y),    # Dummy labels, head is frozen
        "willpower_capacity": np.zeros_like(y)  # Dummy labels, head is frozen
    }

    dataset = tf.data.Dataset.from_tensor_slices((X, y_dict))
    
    if is_training:
        dataset = dataset.shuffle(buffer_size=1000)
        
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

def run_training_pipeline():
    print("=" * 70)
    print("MINDFULLY AI - INITIATING TRAINING PIPELINE")
    print("=" * 70)

    print("\n[1/4] Ingesting EEG Data & Extracting Raw 3D Windows...")
    standardizer = DataStandardizer(target_sfreq=256)
    
    # 1. Resolve paths to the UCI small sample dataset
    path_a, path_c = resolve_sample_paths("uci_small_sample")
    
    # 2. Process Alcoholic Sample (Craving Label = 1)
    print(f"[DataStandardizer] Loading sample A: {path_a}")
    raw_64_a = standardizer.ingest_and_standardize(path_a, apply_spatial_ablation=False)
    raw_fp_a = standardizer.spatial_ablation(raw_64_a, target_nodes=["FP1", "FP2"])
    X_a = extract_raw_windows(raw_fp_a, window_size=1024, step_size=64)
    y_a = np.ones(len(X_a), dtype=np.float32)
    
    # 3. Process Control Sample (Neutral Label = 0)
    print(f"\n[DataStandardizer] Loading sample C: {path_c}")
    raw_64_c = standardizer.ingest_and_standardize(path_c, apply_spatial_ablation=False)
    raw_fp_c = standardizer.spatial_ablation(raw_64_c, target_nodes=["FP1", "FP2"])
    X_c = extract_raw_windows(raw_fp_c, window_size=1024, step_size=64)
    y_c = np.zeros(len(X_c), dtype=np.float32)
    
    # Combine the datasets
    X_fp = np.vstack([X_a, X_c]).astype(np.float32)
    y = np.concatenate([y_a, y_c])

    # Simple 80/20 Train/Validation split
    split_idx = int(len(X_fp) * 0.8)
    X_train, X_val = X_fp[:split_idx], X_fp[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    print(f"\nTraining windows: {len(X_train)} | Validation windows: {len(X_val)}")

    print("\n[2/4] Building tf.data Pipelines...")
    train_dataset = create_tf_dataset(X_train, y_train, batch_size=16, is_training=True)
    val_dataset = create_tf_dataset(X_val, y_val, batch_size=16, is_training=False)

    print("\n[3/4] Initializing and Configuring the Neural Network...")
    model = build_mindfully_model(input_shape=(1024, 2), learning_rate=1e-3)
    
    # APPLY TRANSFER LEARNING: Freeze backbone, train ONLY the substance head
    model = configure_transfer_learning(model, active_head="substance")
    
    # Define callbacks (Early Stopping prevents overfitting to the small dataset)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_substance_urge_loss', 
            patience=5, 
            restore_best_weights=True,
            mode='min'  # Explicitly tell Keras we want this metric to go down
        )
    ]

    print("\n[4/4] Executing Model Training...")
    mlflow.set_experiment("Mindfully_Deep_Learning_Training")
    
    with mlflow.start_run(run_name="substance_head_finetuning"):
        mlflow.log_param("epochs", 20)
        mlflow.log_param("batch_size", 16)
        mlflow.log_param("active_head", "substance_urge")
        
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=20,
            callbacks=callbacks,
            verbose=1
        )
        
        # Log final metrics
        final_val_acc = history.history['val_substance_urge_accuracy'][-1]
        mlflow.log_metric("final_val_accuracy", final_val_acc)
        print(f"\nTraining Complete! Final Validation Accuracy: {final_val_acc * 100:.2f}%")

        # Save the trained model for Phase 5 (Quantization)
        model.save("trained_craving_classifier.keras")
        print("Model saved to: trained_craving_classifier.keras")

if __name__ == "__main__":
    run_training_pipeline()