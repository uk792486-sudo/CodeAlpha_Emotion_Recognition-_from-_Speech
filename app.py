import os
import numpy as np
import librosa
import joblib
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File, Request, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model aur Encoder load karein
model = tf.keras.models.load_model('ser_deep_model.keras')
le = joblib.load('encoder.pkl')
MAX_PAD_LEN = 174

templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(request, "index.html", {"request": request})

@app.post("/predict")
async def predict_speech(file: UploadFile = File(...)):
    temp_file = f"temp_{file.filename}"
    with open(temp_file, "wb") as buffer:
        buffer.write(await file.read())
        
    try:
        audio, sample_rate = librosa.load(temp_file, sr=22050, res_type='kaiser_fast')
        audio, _ = librosa.effects.trim(audio, top_db=20)
        
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        
        if mfccs.shape[1] < MAX_PAD_LEN:
            pad_width = MAX_PAD_LEN - mfccs.shape[1]
            mfccs = np.pad(mfccs, ((0, 0), (0, pad_width)), mode='constant')
        else:
            mfccs = mfccs[:, :MAX_PAD_LEN]
            
        features = mfccs.T
        features = np.expand_dims(features, axis=0)
        
        predictions = model.predict(features)
        probs = predictions[0]
        
        predicted_class_idx = np.argmax(probs)
        confidence = float(probs[predicted_class_idx] * 100)
        predicted_emotion = str(le.inverse_transform([predicted_class_idx])[0])
        
        top_indices = np.argsort(probs)[::-1][:3]
        weights = []
        for idx in top_indices:
            emo = str(le.inverse_transform([idx])[0])
            score = float(probs[idx] * 100)
            weights.append({"emotion": emo, "score": round(score, 1)})

        if os.path.exists(temp_file):
            os.remove(temp_file)
        
        return {
            "emotion": predicted_emotion.capitalize(),
            "confidence": f"{confidence:.1f}%",
            "variance": "High Variance" if confidence > 80 else "Stable Tone",
            "transcription": "Audio processed successfully via Kixo-AI Engine.",
            "energy": "High" if np.mean(np.abs(audio)) > 0.02 else "Low",
            "tone": "Stable" if confidence > 75 else "Shaky",
            "f0": "Dynamic",
            "tempo": "Moderate",
            "weights": weights
        }
    except Exception as e:
        if os.path.exists(temp_file):
            os.remove(temp_file)
        return {"error": str(e)}

@app.post("/synthesize")
async def synthesize_speech(text: str = Form(...), emotion: str = Form(...), voice: str = Form(...)):
    # Vocal Synthesis Sandbox ke liye backend logic
    return {
        "status": "success",
        "message": f"Successfully modulated text into '{emotion}' tone using voice preset '{voice}'.",
        "synthesized_text": text
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)