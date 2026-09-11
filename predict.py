import numpy as np
import librosa
import joblib

# 1. Saved Model Load Karein
model = joblib.load('ser_model.pkl')
print("Model successfully load ho gaya hai!")

# 2. Audio se features extract karne ka function
def predict_audio(file_path):
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccs_scaled = np.mean(mfccs.T, axis=0)
        
        # Model ko 2D array chahiye hoti hai
        features = mfccs_scaled.reshape(1, -1)
        
        # Prediction karein
        prediction = model.predict(features)
        return prediction[0]
    except Exception as e:
        print(f"Error: {e}")
        return None

# 3. Test Audio File ka path
test_file_path = r"archive\TESS Toronto emotional speech set data\YAF_sad\YAF_youth_sad.wav"

result = predict_audio(test_file_path)
print(f"Predicted Emotion: {result}")