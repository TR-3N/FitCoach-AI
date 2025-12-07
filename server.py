import os
from typing import List

import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI
from pydantic import BaseModel

# --------- Load trained model ---------
MODEL_PATH = os.path.join("models", "bicep_model.joblib")
clf = joblib.load(MODEL_PATH)

app = FastAPI(title="FitCoach-AI Bicep Curl API")


# --------- Request models ---------
class Sample(BaseModel):
    time: float
    ax: float
    ay: float
    az: float
    gx: float
    gy: float
    gz: float


class Window(BaseModel):
    samples: List[Sample]


# --------- Feature extraction (same logic as training) ---------
def extract_features_from_window_df(df: pd.DataFrame) -> np.ndarray:
    """
    df columns: time, ax, ay, az, gx, gy, gz
    Returns: 1D numpy array with same feature order as in training.
    """
    features: List[float] = []

    # Basic stats for each axis
    for col in ["ax", "ay", "az", "gx", "gy", "gz"]:
        vals = df[col].values
        features.append(float(np.mean(vals)))
        features.append(float(np.std(vals)))
        features.append(float(np.min(vals)))
        features.append(float(np.max(vals)))
        features.append(float(np.max(vals) - np.min(vals)))

    # Acceleration magnitude features
    ax = df["ax"].values
    ay = df["ay"].values
    az = df["az"].values
    accel_mag = np.sqrt(ax ** 2 + ay ** 2 + az ** 2)
    features.append(float(np.max(accel_mag)))
    features.append(float(np.std(np.diff(accel_mag))))

    # Duration of window
    duration = float(df["time"].iloc[-1] - df["time"].iloc[0])
    features.append(duration)

    return np.array(features, dtype=float).reshape(1, -1)


# --------- API endpoint ---------
@app.post("/classify_window")
def classify_window(window: Window):
    """
    Accept a short IMU window and return CORRECT / INCORRECT with confidence.

    Request JSON format:
    {
      "samples": [
        {"time": 0.0, "ax": ..., "ay": ..., "az": ..., "gx": ..., "gy": ..., "gz": ...},
        ...
      ]
    }
    """
    if not window.samples:
        return {"error": "no samples"}

    data = {
        "time": [s.time for s in window.samples],
        "ax": [s.ax for s in window.samples],
        "ay": [s.ay for s in window.samples],
        "az": [s.az for s in window.samples],
        "gx": [s.gx for s in window.samples],
        "gy": [s.gy for s in window.samples],
        "gz": [s.gz for s in window.samples],
    }
    df = pd.DataFrame(data)

    feats = extract_features_from_window_df(df)
    probs = clf.predict_proba(feats)[0]
    pred = int(clf.predict(feats)[0])
    label = "CORRECT" if pred == 0 else "INCORRECT"
    confidence = float(np.max(probs))

    return {"label": label, "confidence": confidence}
