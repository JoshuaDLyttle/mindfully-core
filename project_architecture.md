<!-- # Mindfully: Project Architecture & Roadmap

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
(For AI Context): "Review this project_architecture.md document in our workspace. Based on our stack and goals, let's build the [Feature Name].""" -->

# Mindfully: Project Architecture & Roadmap

## 1. Project Overview
Mindfully is a proactive neurotechnology platform designed to predict and intercept habit relapse (substance and behavioral) while actively fostering positive habit formation and maintaining healthy routines. It utilizes a sparse-node wearable (Fp1, Fp2, AF7, AF8) and an AI-driven mobile app to deploy personalized Just-In-Time Adaptive Interventions (JITAIs).

## 2. Tech Stack
*   **Frontend Mobile App:** React Native (Expo) for cross-platform deployment, using Redux Toolkit for state management.
*   **Backend & API:** Python 3.11+ (FastAPI) for microservices, backed by PostgreSQL (with pgvector) for relational storage and real-time caching.
*   **Machine Learning Pipeline:** Python, TensorFlow/Keras (for CNN/DAFM architecture), MNE-Python (for EEG preprocessing), and MLflow (for experiment tracking).
*   **AI Development Tools:** 
    *   *Gemini Pro High:* Handles standard code generation, boilerplates, feature ingestion scripts, and utility pipelines (e.g., `DataStandardizer`).
    *   *Gemini Argon:* Utilized specifically for complex deep learning architecture design, advanced mathematical modeling, and soft-thresholding logic.
*   **Data Streaming:** Lab Streaming Layer (LSL) for sub-millisecond hardware-to-software synchronization.
*   **Edge Computing:** TensorFlow Lite (.tflite) for local mobile inference (Successfully quantized to INT8).

## 3. Directory Structure
*   `/ml_pipeline`: Dedicated workspace for data processing, validation, and ML model development. Contains Python training loops, BIDS standardization, soft-threshold denoising, spatial ablation testing scripts, the deep learning architecture (`craving_classifier.py`), and the training loop (`model_trainer.py`).
*   `/edge_models`: Contains exported `.tflite` files (e.g., `mindfully_edge_v1.tflite`) and quantization scripts (`edge_exporter.py`).
*   `/mobile_app`: React Native frontend containing the Working Memory Hijack and Cognitive Negotiation UIs.
*   `/backend` (or `/src`): Contains backend business logic, database schemas, and RESTful APIs serving the mobile interface.

## 4. The Anchor Prompt
(For AI Context): "Review the project_architecture.md file carefully to re-acquire full context on the tech stack, directory layout, and current ML pipeline stage. Based on our stack and goals, let's build the [Feature Name]."

*Note on Anchor Prompt Usage: Use this prompt whenever starting a new session or task. It tells the AI to read this document to regain context, understand component dependencies, and apply design principles before generating new code.*