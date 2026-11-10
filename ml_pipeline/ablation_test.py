"""
Mindfully AI - Spatial Ablation Test Pipeline
---------------------------------------------
Reference: project_architecture.md
- Tech Stack: Python, MNE-Python, scikit-learn, MLflow.
- Goal: Quantify accuracy degradation when downsampling from a clinical 64-channel
  EEG montage to the sparse prefrontal nodes (Fp1, Fp2) specified for the Mindfully
  wearable device.

This script:
1. Imports DataStandardizer to ingest and standardize clinical EEG data (4.0-45.0 Hz filtering).
2. Loads EEG data from uci_small_sample.
3. Extracts sliding window variance features across time.
4. Trains a baseline Random Forest classifier on all 64 channels and logs accuracy to MLflow.
5. Retrains the exact same classifier on prefrontal channels (Fp1, Fp2) and logs accuracy to MLflow.
6. Calculates and outputs the accuracy degradation percentage.
"""

import os
import sys
import argparse
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Silence optional MLflow informational hints
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
import mlflow

from data_standardizer import DataStandardizer


def extract_features_sliding_windows(raw, window_size=64, step_size=16):
    """
    Extracts a basic mock feature (variance) across sliding windows.

    Parameters:
    -----------
    raw : mne.io.Raw
        Standardized MNE Raw EEG object.
    window_size : int
        Window duration in samples (default 64 samples = 0.25 s at 256 Hz).
    step_size : int
        Step / hop size in samples (default 16 samples = 0.0625 s).

    Returns:
    --------
    features : np.ndarray
        Array of shape (n_windows, n_channels) containing variance per channel per window.
    """
    data = raw.get_data()  # shape: (n_channels, n_times)
    n_channels, n_times = data.shape
    features = []

    for start in range(0, n_times - window_size + 1, step_size):
        window = data[:, start : start + window_size]
        # Calculate variance across the time dimension (axis 1) for each channel
        var_feature = np.var(window, axis=1)
        features.append(var_feature)

    if not features:
        raise ValueError(
            f"Recording duration ({n_times} samples) is shorter than window_size ({window_size} samples)."
        )

    return np.array(features)


def resolve_sample_paths(data_dir, sample_a=None, sample_c=None):
    """
    Resolves sample file paths inside the uci_small_sample folder or its subdirectories.
    """
    if sample_a and sample_c:
        if str(sample_c).lower() in ("none", "null", ""):
            return sample_a, None
        return sample_a, sample_c
    elif sample_a and sample_c is None:
        pass  # Proceed to discover or allow single sample if requested

    # Collect candidate search directories
    search_dirs = [
        data_dir,
        os.path.join(os.path.dirname(__file__), data_dir),
        os.path.join(data_dir, "smni_eeg_data"),
        os.path.join(os.path.dirname(__file__), "uci_small_sample"),
        os.path.join(os.path.dirname(__file__), "uci_small_sample", "smni_eeg_data"),
    ]

    all_files = []
    for d in search_dirs:
        if os.path.isdir(d):
            for root, _, files in os.walk(d):
                for f in files:
                    if f.endswith((".tar.gz", ".rd.gz", ".rd")):
                        full_path = os.path.abspath(os.path.join(root, f))
                        if full_path not in all_files:
                            all_files.append(full_path)

    if not all_files:
        raise FileNotFoundError(f"No valid EEG data samples found in or under {data_dir}")

    # Look for pair of alcoholic (a_) and control (c_) sample files
    a_files = [p for p in all_files if os.path.basename(p).startswith("a_")]
    c_files = [p for p in all_files if os.path.basename(p).startswith("c_")]

    if a_files and c_files:
        return a_files[0], c_files[0]
    elif len(all_files) >= 2:
        return all_files[0], all_files[1]
    else:
        return all_files[0], None


def load_dataset(standardizer, data_dir="uci_small_sample", sample_a=None, sample_c=None, window_size=64, step_size=16):
    """
    Loads EEG data samples using DataStandardizer and extracts sliding window variance features.
    Supports paired samples (Alcoholic vs Control) or single sample with simulated binary states.
    """
    path_a, path_c = resolve_sample_paths(data_dir, sample_a, sample_c)

    print(f"\n[DataStandardizer] Loading baseline sample A: {path_a}")
    raw_64_a = standardizer.ingest_and_standardize(path_a, apply_spatial_ablation=False)
    feats_64_a = extract_features_sliding_windows(raw_64_a, window_size=window_size, step_size=step_size)

    # Perform spatial ablation for prefrontal channels (Fp1, Fp2)
    raw_fp_a = standardizer.spatial_ablation(raw_64_a, target_nodes=["FP1", "FP2"])
    feats_fp_a = extract_features_sliding_windows(raw_fp_a, window_size=window_size, step_size=step_size)

    if path_c:
        print(f"\n[DataStandardizer] Loading baseline sample C: {path_c}")
        raw_64_c = standardizer.ingest_and_standardize(path_c, apply_spatial_ablation=False)
        feats_64_c = extract_features_sliding_windows(raw_64_c, window_size=window_size, step_size=step_size)

        raw_fp_c = standardizer.spatial_ablation(raw_64_c, target_nodes=["FP1", "FP2"])
        feats_fp_c = extract_features_sliding_windows(raw_fp_c, window_size=window_size, step_size=step_size)

        X_64 = np.vstack([feats_64_a, feats_64_c])
        X_fp = np.vstack([feats_fp_a, feats_fp_c])
        y = np.concatenate([np.ones(len(feats_64_a), dtype=int), np.zeros(len(feats_64_c), dtype=int)])
    else:
        # Single sample mode: split windows into binary states (e.g. simulated baseline vs activation)
        print("\nSingle sample mode: generating mock binary state labels for sliding windows.")
        X_64 = feats_64_a
        X_fp = feats_fp_a
        n = len(X_64)
        y = np.zeros(n, dtype=int)
        y[n // 2 :] = 1

    channel_names_64 = [ch.upper() for ch in raw_64_a.ch_names]
    channel_names_fp = [ch.upper() for ch in raw_fp_a.ch_names]

    return X_64, X_fp, y, channel_names_64, channel_names_fp


def run_ablation_test(
    data_dir="uci_small_sample",
    sample_a=None,
    sample_c=None,
    window_size=64,
    step_size=16,
    test_size=0.3,
    random_state=42,
    experiment_name="Mindfully_Spatial_Ablation"
):
    """
    Runs the full spatial ablation test:
    1. Ingests and standardizes clinical EEG.
    2. Trains baseline Random Forest classifier on 64 channels; logs to MLflow.
    3. Retrains exact same classifier on prefrontal nodes (Fp1, Fp2); logs to MLflow.
    4. Computes and prints accuracy degradation percentage.
    """
    print("=" * 70)
    print("MINDFULLY AI - SPATIAL ABLATION PIPELINE TEST")
    print("Hardware Target: Prefrontal Sparse-Nodes (Fp1, Fp2)")
    print("=" * 70)

    # 1. Initialize DataStandardizer
    standardizer = DataStandardizer(target_sfreq=256)

    # 2. Ingest and extract sliding window features
    X_64, X_fp, y, ch_64, ch_fp = load_dataset(
        standardizer,
        data_dir=data_dir,
        sample_a=sample_a,
        sample_c=sample_c,
        window_size=window_size,
        step_size=step_size,
    )

    print(f"\nDataset extracted: {X_64.shape[0]} windows across {len(ch_64)} channels.")
    print(f"Prefrontal channels isolated: {ch_fp}")

    # Train / Test Split (stratified on label)
    X_train_64, X_test_64, y_train, y_test = train_test_split(
        X_64, y, test_size=test_size, random_state=random_state, stratify=y
    )
    # Corresponding prefrontal slices
    X_train_fp, X_test_fp, _, _ = train_test_split(
        X_fp, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Configure MLflow
    mlflow.set_experiment(experiment_name)

    # -------------------------------------------------------------
    # 3. Step 1: Baseline Random Forest Classifier (All 64 Channels)
    # -------------------------------------------------------------
    print("\n[Step 1/2] Training Baseline Random Forest on all 64 channels...")
    clf_baseline = RandomForestClassifier(n_estimators=100, random_state=random_state)
    clf_baseline.fit(X_train_64, y_train)

    preds_baseline = clf_baseline.predict(X_test_64)
    acc_baseline = float(accuracy_score(y_test, preds_baseline))

    with mlflow.start_run(run_name="baseline_64_channels"):
        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("channel_count", len(ch_64))
        mlflow.log_param("channels", ", ".join(ch_64))
        mlflow.log_param("feature_type", "variance_sliding_window")
        mlflow.log_param("window_size_samples", window_size)
        mlflow.log_param("step_size_samples", step_size)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("train_samples", len(y_train))
        mlflow.log_param("test_samples", len(y_test))
        mlflow.log_metric("accuracy", acc_baseline)

    print(f"Baseline (64 Channels) Accuracy: {acc_baseline * 100:.2f}% (logged to MLflow)")

    # -------------------------------------------------------------
    # 4. Step 2: Retrain exact same classifier on prefrontal channels (Fp1, Fp2)
    # -------------------------------------------------------------
    print("\n[Step 2/2] Retraining exact same Random Forest on Prefrontal channels (Fp1, Fp2)...")
    clf_prefrontal = RandomForestClassifier(n_estimators=100, random_state=random_state)
    clf_prefrontal.fit(X_train_fp, y_train)

    preds_prefrontal = clf_prefrontal.predict(X_test_fp)
    acc_prefrontal = float(accuracy_score(y_test, preds_prefrontal))

    # Calculate degradation percentage
    if acc_baseline > 0:
        degradation_pct = float(((acc_baseline - acc_prefrontal) / acc_baseline) * 100.0)
    else:
        degradation_pct = 0.0
    absolute_drop = float((acc_baseline - acc_prefrontal) * 100.0)

    with mlflow.start_run(run_name="ablation_prefrontal_fp1_fp2"):
        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("channel_count", len(ch_fp))
        mlflow.log_param("channels", ", ".join(ch_fp))
        mlflow.log_param("feature_type", "variance_sliding_window")
        mlflow.log_param("window_size_samples", window_size)
        mlflow.log_param("step_size_samples", step_size)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("train_samples", len(y_train))
        mlflow.log_param("test_samples", len(y_test))
        mlflow.log_metric("accuracy", acc_prefrontal)
        mlflow.log_metric("accuracy_degradation_pct", degradation_pct)
        mlflow.log_metric("accuracy_absolute_drop_pct", absolute_drop)

    print(f"Prefrontal (Fp1, Fp2) Accuracy:  {acc_prefrontal * 100:.2f}% (logged to MLflow)")

    # -------------------------------------------------------------
    # 5. Output Summary Results
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SPATIAL ABLATION TEST RESULTS SUMMARY")
    print("=" * 70)
    print(f"Baseline (64 Channels) Accuracy        : {acc_baseline * 100:.2f}%")
    print(f"Prefrontal (Fp1, Fp2) Accuracy         : {acc_prefrontal * 100:.2f}%")
    print(f"Accuracy Absolute Drop                 : {absolute_drop:.2f}%")
    print(f"Accuracy Degradation Percentage        : {degradation_pct:.2f}%")
    print("=" * 70)

    return {
        "baseline_accuracy": acc_baseline,
        "prefrontal_accuracy": acc_prefrontal,
        "absolute_drop_pct": absolute_drop,
        "degradation_pct": degradation_pct,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mindfully Spatial Ablation Test Pipeline")
    parser.add_argument("--data_dir", type=str, default="uci_small_sample", help="Path to uci_small_sample directory")
    parser.add_argument("--sample_a", type=str, default=None, help="Path to sample A archive")
    parser.add_argument("--sample_c", type=str, default=None, help="Path to sample C archive")
    parser.add_argument("--window_size", type=int, default=64, help="Window length in samples (default: 64)")
    parser.add_argument("--step_size", type=int, default=16, help="Step size in samples (default: 16)")
    parser.add_argument("--test_size", type=float, default=0.3, help="Test set fraction (default: 0.3)")
    parser.add_argument("--random_state", type=int, default=42, help="Random state seed (default: 42)")
    parser.add_argument("--experiment_name", type=str, default="Mindfully_Spatial_Ablation", help="MLflow experiment name")

    args = parser.parse_args()

    run_ablation_test(
        data_dir=args.data_dir,
        sample_a=args.sample_a,
        sample_c=args.sample_c,
        window_size=args.window_size,
        step_size=args.step_size,
        test_size=args.test_size,
        random_state=args.random_state,
        experiment_name=args.experiment_name,
    )
