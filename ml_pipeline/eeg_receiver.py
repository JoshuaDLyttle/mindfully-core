"""
Mindfully AI - Phase 6: Real-Time EEG Receiver & Edge Inference Engine
======================================================================
File: ml_pipeline/eeg_receiver.py
Reference: project_architecture.md

Objective:
----------
Connect to a real-time prefrontal EEG stream using the Lab Streaming Layer (LSL),
buffer incoming streaming samples into a 4-second sliding window ring buffer (1024
samples at 256 Hz across 2 channels: Fp1, Fp2), and feed the data into our quantized
TensorFlow Lite edge model (mindfully_edge_v1.tflite) via tf.lite.Interpreter.
Outputs the real-time `substance_urge` score to the console to drive Just-In-Time
Adaptive Interventions (JITAIs).
"""

import os
import sys
import time
import argparse
import tempfile
import warnings
import threading
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Callable, Any

# Suppress verbose backend logs and configure caches
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow.lite.*")

import numpy as np
import pylsl
import tensorflow as tf


class EEGRingBuffer:
    """
    Fixed-size circular ring buffer for streaming multi-channel EEG time-series.
    
    Biomedical & Performance Rationale:
    -----------------------------------
    At 256 Hz with 2 channels, continuous allocations and copies of 1024-sample arrays
    cause unnecessary garbage collection pauses on edge devices. This circular buffer
    pre-allocates the memory for a 4-second window (1024 samples) and accepts arbitrary
    chunk lengths from LSL inlets with zero reallocation, maintaining chronological
    order when queried.
    """

    def __init__(self, window_size: int = 1024, num_channels: int = 2):
        self.window_size = window_size
        self.num_channels = num_channels
        self.buffer = np.zeros((window_size, num_channels), dtype=np.float32)
        self.write_idx = 0
        self.total_samples_received = 0
        self.samples_since_last_inference = 0

    def append_chunk(self, chunk: np.ndarray) -> int:
        """
        Appends an incoming 2D chunk of shape (n_samples, num_channels) into the ring buffer.
        """
        chunk = np.asarray(chunk, dtype=np.float32)
        if chunk.ndim == 1:
            chunk = chunk.reshape(-1, 1)

        n_samples = len(chunk)
        if n_samples == 0:
            return 0

        # If incoming chunk exceeds window_size, retain the most recent window_size samples
        if n_samples >= self.window_size:
            self.buffer[:] = chunk[-self.window_size :, : self.num_channels]
            self.write_idx = 0
            self.total_samples_received += n_samples
            self.samples_since_last_inference += n_samples
            return n_samples

        end_idx = self.write_idx + n_samples
        if end_idx <= self.window_size:
            self.buffer[self.write_idx : end_idx] = chunk[:, : self.num_channels]
        else:
            first_part = self.window_size - self.write_idx
            self.buffer[self.write_idx :] = chunk[:first_part, : self.num_channels]
            self.buffer[: end_idx % self.window_size] = chunk[first_part:, : self.num_channels]

        self.write_idx = end_idx % self.window_size
        self.total_samples_received += n_samples
        self.samples_since_last_inference += n_samples
        return n_samples

    def is_full(self) -> bool:
        """Returns True if the buffer has received at least window_size samples."""
        return self.total_samples_received >= self.window_size

    def get_window(self) -> np.ndarray:
        """
        Returns the ordered window of shape (window_size, num_channels)
        from oldest sample (index 0) to newest sample (index window_size - 1).
        """
        if not self.is_full():
            raise ValueError(
                f"Buffer not full yet: {self.total_samples_received}/{self.window_size} samples collected."
            )

        if self.write_idx == 0:
            return self.buffer.copy()

        # Concatenate oldest slice (write_idx to end) with newest slice (0 to write_idx)
        return np.vstack([self.buffer[self.write_idx :], self.buffer[: self.write_idx]])

    def reset_inference_counter(self):
        """Resets the counter of samples received since the last inference invocation."""
        self.samples_since_last_inference = 0


class MindfullyEdgeInference:
    """
    Wrapper for TensorFlow Lite Interpreter loading and edge inference.
    Executes mindfully_edge_v1.tflite with hardware acceleration.
    """

    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"TFLite model file not found at: '{model_path}'. "
                f"Ensure edge_models/edge_exporter.py has been run."
            )

        self.model_path = model_path
        print(f"[Model Loader] Initializing TFLite Interpreter with: {model_path}")
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        # Check for signature runner (preferred modern TFLite API)
        signature_list = self.interpreter.get_signature_list()
        if "serving_default" in signature_list:
            self.runner = self.interpreter.get_signature_runner("serving_default")
            self.signature_mode = True
            print("[Model Loader] Serving signature 'serving_default' attached.")
        else:
            self.runner = None
            self.signature_mode = False
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            print("[Model Loader] Fallback to raw tensor index mapping.")

    def predict(self, window: np.ndarray) -> Dict[str, float]:
        """
        Executes model inference on a 4-second EEG window.

        Parameters:
        -----------
        window : np.ndarray
            Shape (1024, 2) or (1, 1024, 2) containing float32 EEG microvolts.

        Returns:
        --------
        Dict[str, float] with 'substance_urge', 'behavioral_urge', and 'willpower_capacity'.
        """
        if window.ndim == 2:
            window = np.expand_dims(window, axis=0)

        window = window.astype(np.float32)

        if self.signature_mode and self.runner is not None:
            results = self.runner(raw_eeg_window=window)
            return {
                "substance_urge": float(results["substance_urge"][0][0]),
                "behavioral_urge": float(results.get("behavioral_urge", [[0.0]])[0][0]),
                "willpower_capacity": float(results.get("willpower_capacity", [[0.0]])[0][0]),
            }
        else:
            # Fallback for models without signature runner
            self.interpreter.set_tensor(self.input_details[0]["index"], window)
            self.interpreter.invoke()
            outputs = {}
            for out in self.output_details:
                val = float(self.interpreter.get_tensor(out["index"])[0][0])
                name = out["name"].lower()
                if "substance" in name:
                    outputs["substance_urge"] = val
                elif "behavioral" in name:
                    outputs["behavioral_urge"] = val
                elif "willpower" in name:
                    outputs["willpower_capacity"] = val
                else:
                    outputs[out["name"]] = val

            # Ensure substance_urge exists
            if "substance_urge" not in outputs:
                outputs["substance_urge"] = float(
                    self.interpreter.get_tensor(self.output_details[0]["index"])[0][0]
                )
            return outputs


class MockEEGStreamer:
    """
    Generates a synthetic 256 Hz, 2-channel prefrontal EEG stream (Fp1, Fp2) over LSL.
    Useful for testing and verification when physical hardware is not connected.
    """

    def __init__(self, stream_name: str = "Mindfully_Mock_EEG", srate: float = 256.0):
        self.stream_name = stream_name
        self.srate = srate
        self.running = False
        self.thread: Optional[threading.Thread] = None

        info = pylsl.StreamInfo(
            name=self.stream_name,
            type="EEG",
            channel_count=2,
            nominal_srate=self.srate,
            channel_format="float32",
            source_id="mindfully_mock_device_001",
        )
        desc = info.desc()
        channels = desc.append_child("channels")
        ch1 = channels.append_child("channel")
        ch1.append_child_value("label", "FP1")
        ch1.append_child_value("unit", "microvolts")
        ch2 = channels.append_child("channel")
        ch2.append_child_value("label", "FP2")
        ch2.append_child_value("unit", "microvolts")

        self.outlet = pylsl.StreamOutlet(info)

    def _stream_loop(self):
        sample_interval = 1.0 / self.srate
        phase = 0.0
        while self.running:
            # Generate realistic EEG components: Theta (5 Hz), Alpha (10 Hz), Beta (22 Hz) + noise
            t = phase
            theta = 8.0 * np.sin(2.0 * np.pi * 5.0 * t)
            alpha = 12.0 * np.sin(2.0 * np.pi * 10.0 * t)
            beta = 5.0 * np.sin(2.0 * np.pi * 22.0 * t)
            noise_fp1 = np.random.normal(0, 3.0)
            noise_fp2 = np.random.normal(0, 3.0)

            fp1 = float(theta + alpha + beta + noise_fp1)
            fp2 = float(theta * 0.9 + alpha * 1.1 + beta * 0.8 + noise_fp2)

            self.outlet.push_sample([fp1, fp2])
            phase += sample_interval
            time.sleep(sample_interval)

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._stream_loop, daemon=True)
        self.thread.start()
        print(f"[Mock Streamer] Started simulated EEG stream '{self.stream_name}' at {self.srate} Hz.")

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        print("[Mock Streamer] Stopped simulated EEG stream.")


def resolve_channel_indices(info: pylsl.StreamInfo) -> Tuple[List[int], List[str]]:
    """
    Parses LSL stream XML metadata to locate prefrontal channels (FP1, FP2).
    Falls back to channels [0, 1] if explicit labels are absent.
    """
    channel_count = info.channel_count()
    if channel_count < 2:
        raise ValueError(f"EEG stream must have at least 2 channels, but has {channel_count}.")

    labels: List[str] = []
    ch_elem = info.desc().child("channels").child("channel")
    while not ch_elem.empty():
        label = ch_elem.child_value("label")
        if not label:
            label = ch_elem.child_value("name")
        if label:
            labels.append(label.upper())
        ch_elem = ch_elem.next_sibling("channel")

    selected_indices: List[int] = []
    selected_labels: List[str] = []

    # Check for target nodes FP1, FP2
    if "FP1" in labels and "FP2" in labels:
        selected_indices = [labels.index("FP1"), labels.index("FP2")]
        selected_labels = ["FP1", "FP2"]
    else:
        # Default to first 2 channels
        selected_indices = [0, 1]
        selected_labels = labels[:2] if len(labels) >= 2 else ["CH_1", "CH_2"]

    return selected_indices, selected_labels


def format_urge_bar(score: float, width: int = 15) -> str:
    """Formats a visual Unicode progress bar for craving severity."""
    score = max(0.0, min(1.0, score))
    filled = int(round(score * width))
    bar = "█" * filled + "░" * (width - filled)
    return bar


def classify_urge_level(score: float) -> Tuple[str, str]:
    """
    Maps substance urge score into JITAI intervention risk tiers.
    """
    if score >= 0.85:
        return "CRITICAL", "Deploy Working Memory Hijack intervention immediately"
    elif score >= 0.70:
        return "HIGH", "Relapse risk elevated; deploy Cognitive Negotiation"
    elif score >= 0.40:
        return "MODERATE", "Salience elevation detected; monitoring user state"
    else:
        return "LOW", "Baseline homeostasis; no intervention required"


def run_eeg_receiver(
    model_path: str,
    stream_name: Optional[str] = None,
    stream_type: str = "EEG",
    window_size: int = 1024,
    step_size: int = 64,
    timeout: float = 10.0,
    duration: Optional[float] = None,
    mock: bool = False,
    jitai_threshold: float = 0.70,
    on_prediction: Optional[Callable[[Dict[str, Any]], None]] = None,
    stop_event: Optional[threading.Event] = None,
):
    """
    Main loop connecting to LSL, filling the ring buffer, and running edge inference.
    """
    mock_streamer = None
    if mock:
        mock_name = stream_name or "Mindfully_Mock_EEG"
        mock_streamer = MockEEGStreamer(stream_name=mock_name, srate=256.0)
        mock_streamer.start()
        time.sleep(0.5)

    print("=" * 75)
    print(" MINDFULLY AI - REAL-TIME EEG RECEIVER & EDGE INFERENCE")
    print("=" * 75)

    # 1. Initialize TFLite Inference Engine
    infer_engine = MindfullyEdgeInference(model_path=model_path)

    # 2. Discover and Connect to LSL Stream
    print(f"\n[LSL Discovery] Searching for EEG stream (type='{stream_type}', timeout={timeout}s)...")
    if stream_name:
        streams = pylsl.resolve_byprop("name", stream_name, timeout=timeout)
    else:
        streams = pylsl.resolve_byprop("type", stream_type, timeout=timeout)

    if not streams:
        if mock_streamer:
            mock_streamer.stop()
        raise TimeoutError(
            f"No LSL EEG streams found after {timeout} seconds. "
            f"Ensure your EEG headset or streaming daemon is active, or use '--mock' for testing."
        )

    stream_info = streams[0]
    inlet = pylsl.StreamInlet(stream_info, max_buflen=360, max_chunklen=step_size)
    inlet_info = inlet.info()

    # Channel configuration
    ch_indices, ch_labels = resolve_channel_indices(inlet_info)
    nominal_srate = inlet_info.nominal_srate()

    print(f"[LSL Connected] Stream Name    : {inlet_info.name()}")
    print(f"[LSL Connected] Channel Count  : {inlet_info.channel_count()} total (using {ch_labels})")
    print(f"[LSL Connected] Sampling Rate  : {nominal_srate} Hz (Target: 256 Hz)")
    print(f"[LSL Connected] Buffer Settings: Window={window_size} samples (4.0s) | Step={step_size} samples ({step_size/256:.2f}s)")
    print(f"[LSL Connected] JITAI Threshold: {jitai_threshold:.2f}")

    if nominal_srate > 0 and abs(nominal_srate - 256.0) > 1.0:
        print(f"[WARNING] Stream sampling rate ({nominal_srate} Hz) differs from expected 256 Hz.")

    # 3. Setup Ring Buffer
    ring_buffer = EEGRingBuffer(window_size=window_size, num_channels=2)

    print("\n" + "-" * 75)
    print(" STREAMING ACTIVE: Waiting for initial 4-second buffer (1024 samples)...")
    print("-" * 75)

    start_time = time.time()
    inference_count = 0
    jitai_triggers = 0

    try:
        while True:
            # Check external stop event
            if stop_event is not None and stop_event.is_set():
                print("\n[Stream Control] Stop event received. Shutting down EEG receiver loop.")
                break

            # Check duration limit
            if duration and (time.time() - start_time) >= duration:
                print(f"\n[Stream Control] Reached specified duration limit of {duration}s.")
                break

            # Pull multi-sample chunk from LSL
            chunk, timestamps = inlet.pull_chunk(timeout=0.1, max_samples=step_size)
            if not chunk:
                continue

            chunk_arr = np.asarray(chunk, dtype=np.float32)
            # Select channels
            selected_chunk = chunk_arr[:, ch_indices]

            # Ingest into Ring Buffer
            ring_buffer.append_chunk(selected_chunk)

            # Check if buffer is full and step size interval has elapsed
            if ring_buffer.is_full() and ring_buffer.samples_since_last_inference >= step_size:
                # Extract 4-second ordered window (1024, 2)
                eeg_window = ring_buffer.get_window()

                # Run inference through mindfully_edge_v1.tflite
                t_infer_start = time.perf_counter()
                predictions = infer_engine.predict(eeg_window)
                latency_ms = (time.perf_counter() - t_infer_start) * 1000.0

                substance_score = float(predictions["substance_urge"])
                level, action = classify_urge_level(substance_score)
                bar = format_urge_bar(substance_score)
                now_str = datetime.now().strftime("%H:%M:%S.%f")[:-3]

                inference_count += 1
                is_trigger = substance_score >= jitai_threshold
                if is_trigger:
                    jitai_triggers += 1

                trigger_flag = " [! JITAI INTERCEPT !]" if is_trigger else ""
                print(
                    f"[{now_str}] Urge: {substance_score:.4f} | [{bar}] | "
                    f"Level: {level:<8} | Latency: {latency_ms:4.1f}ms{trigger_flag}"
                )

                # Broadcast hook for orchestrator or external consumer
                if on_prediction is not None:
                    try:
                        on_prediction({
                            "type": "CRAVING_SCORE",
                            "score": substance_score,
                            "level": level,
                            "latency_ms": latency_ms,
                            "timestamp": now_str,
                            "is_trigger": is_trigger,
                            "all_predictions": predictions,
                        })
                    except Exception as cb_err:
                        print(f"[Warning] Error in on_prediction callback: {cb_err}")

                ring_buffer.reset_inference_counter()

    except KeyboardInterrupt:
        print("\n[Stream Control] Stream acquisition interrupted by user (Ctrl+C).")
    finally:
        try:
            inlet.close_stream()
        except Exception:
            pass

        if mock_streamer is not None:
            mock_streamer.stop()

        elapsed = time.time() - start_time
        print("\n" + "=" * 75)
        print("MINDFULLY RECEIVER SESSION SUMMARY")
        print("=" * 75)
        print(f"Elapsed Time           : {elapsed:.2f} seconds")
        print(f"Total Samples Received : {ring_buffer.total_samples_received}")
        print(f"Total Inferences Run   : {inference_count}")
        print(f"JITAI Threshold Alerts : {jitai_triggers}")
        if inference_count > 0:
            print(f"Average Ingestion Rate : {ring_buffer.total_samples_received / elapsed:.1f} samples/sec")
        print("=" * 75)


def parse_arguments():
    default_model_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "edge_models",
        "mindfully_edge_v1.tflite",
    )

    parser = argparse.ArgumentParser(
        description="Mindfully AI - Real-Time EEG LSL Receiver and Edge TFLite Inference"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=default_model_path,
        help=f"Path to .tflite model file (default: {default_model_path})",
    )
    parser.add_argument(
        "--stream-name",
        type=str,
        default=None,
        help="Optional specific LSL EEG stream name to match.",
    )
    parser.add_argument(
        "--stream-type",
        type=str,
        default="EEG",
        help="LSL stream type property to match (default: 'EEG').",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=1024,
        help="Sliding window size in samples (default: 1024 = 4.0s at 256 Hz).",
    )
    parser.add_argument(
        "--step-size",
        type=int,
        default=64,
        help="Step / hop size in samples between inferences (default: 64 = 0.25s).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="Timeout in seconds when searching for LSL streams (default: 10.0s).",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Optional run duration in seconds before auto-exiting (useful for tests).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.70,
        help="Substance urge score threshold for JITAI alerts (default: 0.70).",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Launch an internal simulated 2-channel 256 Hz EEG stream for testing.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    run_eeg_receiver(
        model_path=args.model_path,
        stream_name=args.stream_name,
        stream_type=args.stream_type,
        window_size=args.window_size,
        step_size=args.step_size,
        timeout=args.timeout,
        duration=args.duration,
        mock=args.mock,
        jitai_threshold=args.threshold,
    )

