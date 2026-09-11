import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Bidirectional
from tensorflow.keras.utils import to_categorical
import joblib

DATASET_PATH = r"archive\TESS Toronto emotional speech set data"
MAX_PAD_LEN = 174  # Audio files ko aik jaisi length dene ke liye

def extract_features(file_path):
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        
        # Padding ya truncation taake sabhi audio tensors ka size aik jaisa ho
        if mfccs.shape[1] < MAX_PAD_LEN:
            pad_width = MAX_PAD_LEN - mfccs.shape[1]
            mfccs = np.pad(mfccs, ((0, 0), (0, pad_width)), mode='constant')
        else:
            mfccs = mfccs[:, :MAX_PAD_LEN]
            
        return mfccs.T  # Shape: (time_steps, n_mfcc)
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return None

def load_data(dataset_path):
    features = []
    labels = []
    
    if not os.path.exists(dataset_path):
        print(f"Directory {dataset_path} nahi mili!")
        return np.array([]), np.array([])

    print("Dataset load aur feature extraction ho rahi hai...")
    for emotion in os.listdir(dataset_path):
        emotion_dir = os.path.join(dataset_path, emotion)
        if os.path.isdir(emotion_dir):
            for file in os.listdir(emotion_dir):
                if file.endswith('.wav'):
                    file_path = os.path.join(emotion_dir, file)
                    data = extract_features(file_path)
                    if data is not None:
                        features.append(data)
                        labels.append(emotion)
                        
    return np.array(features), np.array(labels)

# 1. Data Load Karein
X, y = load_data(DATASET_PATH)

if len(X) > 0:
    # Labels ko numbers mein encode karein
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    y_categorical = to_categorical(y_encoded)
    
    # Save Label Encoder taake prediction ke waqt kaam aaye
    joblib.dump(le, 'encoder.pkl')

    # Train-Test Split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)

    print("Deep Learning LSTM Model train ho raha hai...")
    
    # 2. LSTM Deep Learning Architecture
    model = Sequential([
        Bidirectional(LSTM(128, return_sequences=True), input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.3),
        Bidirectional(LSTM(64)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(y_categorical.shape[1], activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # 3. Model Training
    history = model.fit(X_train, y_train, epochs=25, batch_size=32, validation_data=(X_test, y_test))

    # 4. Model Save Karein
    model.save('ser_deep_model.keras')
    print("Deep Learning LSTM Model successfully 'ser_deep_model.keras' ke tor par save ho gaya hai!")
else:
    print("Dataset khali hai ya path theek nahi hai.")
