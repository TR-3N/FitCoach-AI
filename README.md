# FitCoach-AI – Bicep Curl Form Coach

FitCoach-AI is a small end-to-end system that uses phone motion sensors (accelerometer + gyroscope) and a machine learning model to give **real-time** feedback on bicep curl form. It tells you whether a rep looks **CORRECT** or **INCORRECT** and keeps basic rep statistics.

> ⚠️ **Scope:** Right now this project only supports **bicep curls**. The architecture is designed so more exercises (squats, push-ups, etc.) can be added later with new models and UI updates.

***

## Project structure

- `train_bicep_model.py`, `classify_new_session.py`  
  - Python scripts to train a RandomForest classifier from IMU CSVs and classify new sessions into CORRECT vs INCORRECT windows.
- `models/bicep_model.joblib`  
  - Trained model used by the backend API.  
- `server.py`  
  - FastAPI backend that exposes `POST /classify_window` and returns `{ "label": "CORRECT" | "INCORRECT", "confidence": float }` for a window of IMU samples.
- `android_app/`  
  - Android client (Kotlin) that reads accelerometer/gyroscope data, buffers it into sliding windows, sends them to the backend, and shows:
    - Live status (CORRECT / INCORRECT).  
    - Total reps, correct reps, incorrect reps in a simple fitness-style UI.

***

## How to run (local dev)

1. **Backend (FastAPI + model)**  
   - Install Python dependencies.  
   - Make sure `models/bicep_model.joblib` exists (run `python train_bicep_model.py` if needed).  
   - Start the API:

     ```bash
     uvicorn server:app --host 0.0.0.0 --port 8000 --reload
     ```

   - Open `http://127.0.0.1:8000/docs` to see the API docs.

2. **Android app (phone + sensors)**  
   - Open `android_app` in Android Studio.  
   - Update the base URL in `MainActivity.kt`:

     ```kotlin
     private const val BASE_URL = "http://YOUR_LAPTOP_IP:8000"
     ```

     (Laptop and phone must be on the same Wi‑Fi network.)
   - Ensure `network_security_config.xml` allows HTTP to this IP for local development.  
   - Run the app on a real Android device, tap **Start**, and perform bicep curls with the phone/watch on your training arm.

***

## How it works

- The Android app collects IMU samples, keeps a sliding 3‑second window, and periodically sends a window to the backend when:  
  - it has enough samples,  
  - there is enough change in acceleration to look like a rep, and  
  - at least ~1–1.5 seconds have passed since the last counted rep.
- The FastAPI service converts the window into features and calls the trained RandomForest model, which outputs CORRECT vs INCORRECT and a confidence score.  
- The app updates:
  - **Status text** (CORRECT / INCORRECT + confidence).  
  - **Rep counters** (total, correct, incorrect).

***

## Future work

Planned extensions:

- Add more exercises (e.g., squats, push-ups) by training additional models or extending labels (BICEP_CORRECT, SQUAT_INCORRECT, etc.).
- Per-user calibration and better rep segmentation (per‑rep peak detection instead of simple sliding windows).  
- More advanced UI with history, sets, and progress tracking similar to commercial fitness apps.

***
