from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import uvicorn
import librosa
import numpy as np
import joblib
from pydub import AudioSegment
import io

# Load trained model
MODEL_PATH = "features/als_classifier.pkl"
clf = joblib.load(MODEL_PATH)

app = FastAPI(title="ALS Audio Classifier")
templates = Jinja2Templates(directory="templates")

# Audio preprocessing
def load_audio(file_bytes, sr=16000):
    try:
        # Try librosa first
        y, _ = librosa.load(io.BytesIO(file_bytes), sr=sr, mono=True)
    except Exception:
        try:
            # Fallback to pydub
            audio = AudioSegment.from_file(io.BytesIO(file_bytes))
            audio = audio.set_frame_rate(sr).set_channels(1)
            samples = np.array(audio.get_array_of_samples()).astype(np.float32)
            y = samples / np.iinfo(audio.array_type).max
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Could not decode audio: {str(e)}")
    # Remove silence
    if len(y) > 0:
        y, _ = librosa.effects.trim(y)
        # Normalize
        if len(y) > 0:
            y = librosa.util.normalize(y)
        else:
            raise HTTPException(status_code=400, detail="Audio too short after trimming silence.")
    else:
        raise HTTPException(status_code=400, detail="Audio file is empty.")
    return y

# Feature extraction
def extract_mfcc(y, sr=16000, n_mfcc=13):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc.T, axis=0)



@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".wav", ".mp3", ".flac", ".m4a", ".ogg")):
        raise HTTPException(status_code=400, detail="Unsupported file type.")
    
    # Read audio bytes and extract features
    audio_bytes = await file.read()
    y = load_audio(audio_bytes)
    mfcc = extract_mfcc(y)
    X = mfcc.reshape(1, -1)
    
    # Get prediction and probabilities
    pred = clf.predict(X)[0]
    proba = clf.predict_proba(X)[0]  # array: [Control_prob, ALS_prob]
    
    # Map class number to label
    label_map = {0: "Control", 1: "ALS"}
    label = label_map[pred]
    
    # Prepare probabilities as percentage
    proba_dict = {
        "Control": round(float(proba[0]) * 100, 2),
        "ALS": round(float(proba[1]) * 100, 2)
    }
    
    return JSONResponse(content={"prediction": label, "probability": proba_dict})


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
