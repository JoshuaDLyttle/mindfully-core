"""
Mindfully AI - DataStandardizer Pipeline
----------------------------------------
This script ingests clinical EEG datasets, applies temporal bandpass filtering 
(explicitly skipping spatial ICA), and performs spatial downsampling to isolate 
only the prefrontal nodes (Fp1, Fp2, AF7, AF8) required for the wearable.
"""

import mne
import numpy as np

class DataStandardizer:
    def __init__(self, target_sfreq=256):
        # Standardize all incoming data to 256 Hz
        self.target_sfreq = target_sfreq
        
        # TARGET HARDWARE NODES: 
        # Fp1/Fp2 (Frontopolar), AF7/AF8 (Anterior Frontal) 
        # Plus a common reference node (e.g., Cz, A1, A2, or mastoids)
        # Note: We use uppercase strings to account for varying dataset nomenclatures
        self.target_nodes = ['FP1', 'FP2', 'AF7', 'AF8', 'CZ', 'A1', 'A2', 'TP9', 'TP10']

    def ingest_and_standardize(self, file_path):
        """
        Loads the raw EEG file, applies temporal filters, and drops non-target channels.
        """
        print(f"Loading EEG data from: {file_path}")
        
        # 1. INGESTION
        # Uses MNE to read standard neuro-formats (EDF, VHDR, BDF, etc.)
        raw = mne.io.read_raw(file_path, preload=True)

        # Resample if the dataset's frequency doesn't match our 256Hz target
        if raw.info['sfreq'] != self.target_sfreq:
            print(f"Resampling data from {raw.info['sfreq']}Hz to {self.target_sfreq}Hz...")
            raw.resample(self.target_sfreq)

        # 2. TEMPORAL FILTERING (The "No-ICA" Approach)
        print("Applying 4.0 - 45.0 Hz zero-phase Butterworth bandpass filter...")
        # BIOMEDICAL NOTE: We explicitly drop ICA here because 2 nodes cannot triangulate space.
        # 4.0 Hz high-pass removes sweat/movement drifts.
        # 45.0 Hz low-pass removes 50/60 Hz power-line noise from wall outlets.
        raw.filter(l_freq=4.0, h_freq=45.0, phase='zero', fir_design='firwin')

        # 3. SPATIAL DOWNSAMPLING (Hardware Simulation)
        print("Performing spatial ablation (dropping non-prefrontal channels)...")
        # Standardize channel names to uppercase for matching
        current_channels = [ch.upper() for ch in raw.ch_names]
        
        # Find which of our target nodes actually exist in this specific dataset
        channels_to_keep = [
            raw.ch_names[i] for i, ch in enumerate(current_channels) if ch in self.target_nodes
        ]
        
        if not channels_to_keep:
            raise ValueError("None of the target prefrontal nodes were found in this dataset!")

        # Mathematically drop all 50+ other channels, keeping only our targets
        raw.pick_channels(channels_to_keep)
        print(f"Successfully ablated down to {len(raw.ch_names)} channels: {raw.ch_names}")
        
        return raw

# ==========================================
# Quick Execution Test
# ==========================================
if __name__ == "__main__":
    # When we write the ablation_test.py, it will call this class.
    print("DataStandardizer initialized and ready for spatial ablation tests.")