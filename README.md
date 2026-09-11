# 🎙️ Speech Emotion Recognition Dashboard (CodeAlpha Task 2)

An end-to-end **Speech Emotion Recognition (SER)** web application developed as **Task 2** of my Artificial Intelligence/Python Internship at **CodeAlpha**. The dashboard features a high-end **VIP Classic Gold & Obsidian Dark** glassmorphism interface that analyzes human speech, tracks acoustic confidence metrics, and displays real-time emotional fingerprint weights.

---

## 🌟 Key Features

* **Live Audio Recording:** Built-in microphone recorder using the browser's native `MediaRecorder` API with a live session timer and custom gold audio waveform visualizer.
* **Audio File Upload Support:** Ability to upload pre-recorded audio tracks (`.wav`, `.mp3`, `.ogg`) for instant classification.
* **Acoustic Metrics Analysis:** Evaluates primary detected emotions along with confidence coefficients, acoustic energy, tone stability, F0 prosody, and tempo.
* **Emotional Fingerprint Weights:** Dynamic progress bars tracking secondary emotions (e.g., Sad, Fearful, Neutral).

---

## 🛠️ Tech Stack

* **Frontend:** HTML5, CSS3, JavaScript (Vanilla ES6+), Custom Glassmorphism UI, Plus Jakarta Sans font.
* **Backend:** Python, Flask / FastAPI backend architecture.
* **Environment Management:** Miniconda (`ser_env`).

---

## 🚀 Setup & Installation Guide (VS Code)

To set up and run this project locally on your machine (e.g., saved in drive `C:`), follow these steps:

1. **Open the Project in VS Code:**
   * Launch Visual Studio Code and open your project folder (e.g., located at `C:\CodeAlpha_Emotion_Recognition_from_Speech`).

2. **Set Up the Python Environment (Miniconda):**
   * Open the integrated terminal in VS Code (`Ctrl + ~`).
   * Verify or activate your Miniconda virtual environment (`ser_env`). If you need to create it from scratch:
     ```bash
     conda create -n ser_env python=3.10
     conda activate ser_env
     ```

3. **Install Required Dependencies:**
   * Install the necessary Python packages (such as Flask/FastAPI, Librosa/audio processing libraries, etc.) using pip:
     ```bash
     pip install -r requirements.txt
     ```

---

## ▶️ Running the Application

To run the application using your specific Miniconda python environment path, execute the following command in your VS Code terminal:

```bash
C:\miniconda3\envs\ser_env\python.exe app.py