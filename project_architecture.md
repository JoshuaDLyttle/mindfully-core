# Mindfully: Project Architecture & Roadmap

## 1. Project Overview
Mindfully is a proactive neurotechnology platform that predicts and intercepts habit relapse (substance and behavioral) and supports positive habit formation using real-time EEG data. 
It utilizes a sparse-node wearable (Fp1, Fp2, AF7, AF8) and an AI-driven mobile app to deploy Just-In-Time Adaptive Interventions (JITAIs).

## 2. Tech Stack
*   **Backend & Machine Learning:** Python, TensorFlow/Keras (for CNN/DAFM architecture), MNE-Python (for EEG preprocessing), MLflow (for experiment tracking).
*   **Data Streaming:** Lab Streaming Layer (LSL) for sub-millisecond hardware-to-software synchronization.
*   **Frontend Mobile App:** React Native (Expo).
*   **Edge Computing:** TensorFlow Lite (.tflite) for local mobile inference.

## 3. Directory Structure
*   `/ml_pipeline`: Python training loops, BIDS standardization, soft-threshold denoising, and spatial ablation testing scripts.
*   `/edge_models`: Exported `.tflite` files and quantization scripts.
*   `/mobile_app`: React Native frontend containing the Working Memory Hijack and Cognitive Negotiation UIs.

## 4. The Anchor Prompt
(For AI Context): "Review this project_architecture.md document in our workspace. Based on our stack and goals, let's build the [Feature Name]."