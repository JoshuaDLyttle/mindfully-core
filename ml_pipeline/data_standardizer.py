"""
Mindfully AI - DataStandardizer Pipeline
----------------------------------------
This script ingests clinical EEG datasets, applies temporal bandpass filtering 
(explicitly skipping spatial ICA), and performs spatial downsampling to isolate 
only the prefrontal nodes (Fp1, Fp2, AF7, AF8) required for the wearable.
"""

import os
import gzip
import tarfile
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

    def _parse_uci_text(self, content):
        """
        Parses raw text from UCI EEG database (.rd format).
        Lines have format: <trial_num> <channel_name> <sample_idx> <voltage_val>
        """
        lines = content.splitlines()
        data_dict = {}
        for line in lines:
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) == 4:
                ch, val = parts[1].upper(), float(parts[3])
                if ch not in data_dict:
                    data_dict[ch] = []
                data_dict[ch].append(val)
        ch_names = list(data_dict.keys())
        data = np.array([data_dict[c] for c in ch_names], dtype=np.float64)
        return ch_names, data

    def _load_uci_data(self, file_path):
        """
        Loads EEG trials from a UCI .tar.gz archive, .rd.gz, or .rd file into an MNE RawArray.
        """
        trials_data = []
        ch_names = None

        if tarfile.is_tarfile(file_path):
            with tarfile.open(file_path) as tf:
                for member in sorted(tf.getmembers(), key=lambda m: m.name):
                    if not member.name.endswith(('.gz', '.rd')):
                        continue
                    extracted = tf.extractfile(member)
                    if member.name.endswith('.gz'):
                        with gzip.GzipFile(fileobj=extracted) as gz:
                            content = gz.read().decode('utf-8', errors='ignore')
                    else:
                        content = extracted.read().decode('utf-8', errors='ignore')
                    chs, arr = self._parse_uci_text(content)
                    if ch_names is None:
                        ch_names = chs
                    trials_data.append(arr)
        elif str(file_path).endswith('.gz'):
            with gzip.open(file_path, 'rt', errors='ignore') as gz:
                chs, arr = self._parse_uci_text(gz.read())
                ch_names = chs
                trials_data.append(arr)
        else:
            with open(file_path, 'r', errors='ignore') as f:
                chs, arr = self._parse_uci_text(f.read())
                ch_names = chs
                trials_data.append(arr)

        if not trials_data:
            raise ValueError(f"No valid EEG trials extracted from {file_path}")

        # Concatenate trials along time axis (axis 1)
        full_data = np.concatenate(trials_data, axis=1)
        info = mne.create_info(ch_names=ch_names, sfreq=self.target_sfreq, ch_types='eeg')
        return mne.io.RawArray(full_data, info, verbose=False)

    def load_raw(self, file_path):
        """
        Loads raw EEG data from either standard neuro formats or UCI formats.
        """
        print(f"Loading EEG data from: {file_path}")
        if str(file_path).endswith(('.tar.gz', '.rd.gz', '.rd')) or (os.path.isfile(file_path) and tarfile.is_tarfile(file_path)):
            return self._load_uci_data(file_path)
        return mne.io.read_raw(file_path, preload=True)

    def spatial_ablation(self, raw, target_nodes=None):
        """
        Performs spatial ablation by retaining only the specified target nodes.
        """
        if target_nodes is None:
            target_nodes = self.target_nodes
        target_nodes_upper = [n.upper() for n in target_nodes]
        print(f"Performing spatial ablation (retaining {target_nodes_upper})...")

        current_channels = [ch.upper() for ch in raw.ch_names]
        channels_to_keep = [
            raw.ch_names[i] for i, ch in enumerate(current_channels) if ch in target_nodes_upper
        ]

        if not channels_to_keep:
            raise ValueError(f"None of target nodes {target_nodes_upper} found in dataset!")

        raw_ablated = raw.copy().pick(channels_to_keep)
        print(f"Successfully ablated down to {len(raw_ablated.ch_names)} channels: {raw_ablated.ch_names}")
        return raw_ablated

    def ingest_and_standardize(self, file_path, apply_spatial_ablation=True, target_nodes=None):
        """
        Loads the raw EEG file, applies temporal filters, and optionally drops non-target channels.
        """
        # 1. INGESTION
        raw = self.load_raw(file_path)

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
        if apply_spatial_ablation:
            raw = self.spatial_ablation(raw, target_nodes=target_nodes)
        
        return raw

# ==========================================
# Quick Execution Test
# ==========================================
if __name__ == "__main__":
    # When we write the ablation_test.py, it will call this class.
    print("DataStandardizer initialized and ready for spatial ablation tests.")