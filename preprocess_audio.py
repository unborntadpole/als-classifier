import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import librosa
from pydub import AudioSegment


# -------------------------
# 1. Safe audio loader
# -------------------------
def safe_load(file_path, sr=16000):
    """Load audio safely with librosa, fallback to pydub if needed. 
    Skip file if unreadable."""
    try:
        y, _ = librosa.load(file_path, sr=sr, mono=True)
        return y
    except Exception as e1:
        print(f"[WARN] Librosa failed on {file_path}, trying pydub. Error: {e1}")
        try:
            sound = AudioSegment.from_file(file_path)
            sound = sound.set_frame_rate(sr).set_channels(1)
            samples = np.array(sound.get_array_of_samples()).astype(np.float32)
            y = samples / np.iinfo(sound.array_type).max
            return y
        except Exception as e2:
            print(f"[ERROR] Skipping {file_path}. Could not decode. Error: {e2}")
            return None


# -------------------------
# 2. Preprocessing functions
# -------------------------
def load_and_preprocess(file_path, sr=16000):
    # Load audio safely
    y = safe_load(file_path, sr=sr)
    
    # Skip if audio could not be loaded
    if y is None:
        return None

    # Trim leading/trailing silence
    y, _ = librosa.effects.trim(y)
    
    # Skip if trimmed audio is empty
    if y.size == 0:
        return None
    
    # Normalize loudness
    y = librosa.util.normalize(y)
    
    return y


def extract_mfcc(y, sr=16000, n_mfcc=13):
    """Return mean MFCC feature vector"""
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=512)  # smaller FFT for short clips
    return np.mean(mfcc.T, axis=0)


def extract_melspectrogram(y, sr=16000, n_mels=128):
    """Return log-mel spectrogram (2D array)"""
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db


# -------------------------
# 3. Load metadata CSV
# -------------------------
# Assuming tab-separated: file_path \t label
df = pd.read_csv("dataset_metadata.csv")

# -------------------------
# 4. Feature extraction loop
# -------------------------
mfcc_features = []
labels = []

for _, row in tqdm(df.iterrows(), total=len(df)):
    y = load_and_preprocess(row["file"])
    if y is None:   # skip unreadable files
        continue

    mfcc = extract_mfcc(y)
    mfcc_features.append(mfcc)
    labels.append(1 if row["label"] == "ALS" else 0)

# Convert to numpy arrays
X = np.array(mfcc_features)
y = np.array(labels)

print("✅ Feature matrix:", X.shape)
print("✅ Labels:", y.shape)

# -------------------------
# 5. Save for later use
# -------------------------
os.makedirs("features", exist_ok=True)
np.save("features/X_mfcc.npy", X)
np.save("features/y_labels.npy", y)
