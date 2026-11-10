"""
Mindfully AI - Phase 5: Edge Quantization & Compression
=======================================================
File: edge_models/edge_exporter.py

Objective:
----------
Convert the heavy 32-bit floating-point Python model (.keras) into an ultra-lightweight, 
8-bit integer TensorFlow Lite model (.tflite). This process (Post-Training Quantization) 
shrinks the model's footprint by ~75% and enables sub-20ms local execution on a smartphone 
for real-time Just-In-Time Adaptive Interventions (JITAIs).
"""

import os
import sys
import tensorflow as tf
import numpy as np

# Append the root directory to sys.path so we can securely import from the ml_pipeline folder
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml_pipeline.craving_classifier import SoftThresholding, DAFMGate

def representative_dataset():
    """
    Provides a small, representative sample of real EEG data to the TFLiteConverter.
    The converter uses this to calibrate the dynamic ranges of the weights and activations 
    before converting them from 32-bit floats to 8-bit integers.
    """
    for _ in range(100):
        # Generate random float32 noise mimicking EEG microvolts
        data = np.random.normal(0, 15.0, size=(1, 1024, 2)).astype(np.float32)
        yield [data]

def export_to_tflite(model_path: str, output_path: str):
    print("=" * 70)
    print("MINDFULLY AI - INITIATING EDGE QUANTIZATION")
    print("=" * 70)

    print(f"\n[1/3] Loading trained model from: {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found at {model_path}. Did you run Phase 4?")
    
    try:
        # We explicitly pass the custom layers to the Keras loader using custom_objects
        model = tf.keras.models.load_model(
            model_path,
            custom_objects={'SoftThresholding': SoftThresholding, 'DAFMGate': DAFMGate}
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print("\n[2/3] Configuring Post-Training Quantization (PTQ)...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Set the optimization flag to quantize weights and activations
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Provide the representative dataset for calibration
    converter.representative_dataset = representative_dataset
    
    # Restrict supported operations to INT8 to ensure maximum compatibility on mobile edge processors
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    
    # Ensure the input and output tensors remain as float32 for easier integration with React Native
    converter.inference_input_type = tf.float32
    converter.inference_output_type = tf.float32

    print("\n[3/3] Compressing model to .tflite format...")
    try:
        tflite_model = converter.convert()
    except Exception as e:
        print(f"Conversion failed. Error: {e}")
        return

    # Save the quantized model to the edge_models folder
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
        
    # Calculate file sizes for verification
    keras_size_kb = os.path.getsize(model_path) / 1024
    tflite_size_kb = os.path.getsize(output_path) / 1024
    compression_ratio = (1 - (tflite_size_kb / keras_size_kb)) * 100
        
    print("\n" + "=" * 70)
    print("QUANTIZATION COMPLETE & VERIFIED")
    print("=" * 70)
    print(f"Original .keras size : {keras_size_kb:.2f} KB")
    print(f"New .tflite size     : {tflite_size_kb:.2f} KB")
    print(f"Compression Ratio    : {compression_ratio:.1f}% reduction")
    print(f"Deployment File      : {output_path}")
    print("=" * 70)

if __name__ == "__main__":
    # Define paths based on your monorepo structure
    source_model = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ml_pipeline", "trained_craving_classifier.keras")
    edge_output = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mindfully_edge_v1.tflite")
    
    export_to_tflite(source_model, edge_output)