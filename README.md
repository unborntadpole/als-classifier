# ALS Audio Classifier

A Python-based machine learning project that detects **Amyotrophic Lateral Sclerosis (ALS)** from voice recordings using audio features and a classifier model. The project includes audio preprocessing, feature extraction, model training, and a web application for inference.

---

## Table of Contents

- [Project Overview](#project-overview)  
- [Dataset](#dataset)  
- [Audio Preprocessing](#audio-preprocessing)  
- [Feature Extraction](#feature-extraction)  
- [Model Training](#model-training)  
- [Web Application](#web-application)  
- [Tech Stack](#tech-stack)

---

## Project Overview

The goal of this project is to classify whether a person’s voice recording indicates ALS or is healthy (Control). The workflow consists of:

1. Preprocessing raw audio recordings.
2. Extracting relevant audio features (MFCCs, Mel-spectrograms).
3. Training a classifier model (Random Forest) using the extracted features.
4. Deploying the model with a FastAPI web application for online predictions.

---

## Dataset

- **Source:** Processed dataset collected from various ALS patients and healthy controls.
- **Format:** Each sample is a WAV file with its corresponding label (`ALS` or `Control`).
- **Metadata:** A CSV file (`dataset_metadata.csv`) contains the following columns:  
  - `file`: path to the audio file  
  - `label`: target label (`ALS` or `Control`)  

---

## Audio Preprocessing

- **Resampling:** All audio files are resampled to 16 kHz mono.  
- **Trimming:** Leading and trailing silence is removed using librosa’s `trim` function.  
- **Normalization:** Audio signals are normalized for consistent loudness.  
- **Error Handling:** Unsupported or corrupted audio files are skipped safely.
- ⚠️ **Warning: FFmpeg must be installed externally on your system for audio decoding to work.**

---

## Feature Extraction

Two main feature extraction techniques are used:

1. **MFCC (Mel Frequency Cepstral Coefficients):**  
   - 13 coefficients per frame  
   - Mean aggregation across frames to obtain fixed-size feature vector  

2. **Mel-Spectrogram (Optional, 2D features):**  
   - 128 Mel bands  
   - Log-scaled for stability  

The extracted features are saved as NumPy arrays:

- `features/X_mfcc.npy` → Feature matrix  
- `features/y_labels.npy` → Labels  

---

## Model Training

- **Model:** Random Forest Classifier (sklearn)  
- **Training/Test Split:** 80/20  
- **Evaluation Metrics:** Accuracy, precision, recall, F1-score  

**Performance on test set:**
- Accuracy: 0.91
- Precision (ALS/Control): 0.92 / 0.90
- Recall (ALS/Control): 0.80 / 0.96
- F1-score (ALS/Control): 0.86 / 0.93


- The trained model is saved as `features/als_classifier.pkl`.

---

## Web Application

A **FastAPI** web application allows users to upload audio files and get real-time ALS predictions:

- **Endpoint:** `/predict`  
- **Input:** WAV, MP3, FLAC, M4A, or OGG audio file  
- **Output:** JSON response with prediction label and probability  


---

## Tech Stack

- **Python 3.11**
- **Libraries:**
  - `librosa` → Audio processing and feature extraction  
  - `pydub` → Audio format handling  
  - `numpy`, `pandas` → Data handling  
  - `scikit-learn` → Model training and evaluation  
  - `FastAPI` → API server  
  - `Uvicorn` → ASGI server  
- **Frontend:** HTML + Vanilla JavaScript (no Bootstrap)
