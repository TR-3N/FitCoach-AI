import os
import sys
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import joblib

# Must match train_bicep_model.py
ACC_FILE = "Accelerometer.csv"
GYRO_FILE = "Gyroscope.csv"


def load_and_merge(acc_path, gyro_path, resample_dt=0.01):
    acc = pd.read_csv(acc_path)
    gyro = pd.read_csv(gyro_path)

    acc = acc.rename(columns={
        acc.columns[0]: "time",
        acc.columns[1]: "ax",
        acc.columns[2]: "ay",
        acc.columns[3]: "az",
    })
    gyro = gyro.rename(columns={
        gyro.columns[0]: "time",
        gyro.columns[1]: "gx",
        gyro.columns[2]: "gy",
        gyro.columns[3]: "gz",
    })

    acc["time"] = acc["time"].astype(float)
    gyro["time"] = gyro["time"].astype(float)

    acc = acc.drop_duplicates(subset="time", keep="first")
    gyro = gyro.drop_duplicates(subset="time", keep="first")

    t_start = max(acc["time"].min(), gyro["time"].min())
    t_end = min(acc["time"].max(), gyro["time"].max())
    times = np.arange(t_start, t_end, resample_dt)

    acc_idx = acc.set_index("time").reindex(times)
    gyro_idx = gyro.set_index("time").reindex(times)

    acc_interp = acc_idx.interpolate(method="linear").ffill().bfill()
    gyro_interp = gyro_idx.interpolate(method="linear").ffill().bfill()

    df = pd.DataFrame(index=times)
    df["time"] = times
    df["ax"] = acc_interp["ax"].values
    df["ay"] = acc_interp["ay"].values
    df["az"] = acc_interp["az"].values
    df["gx"] = gyro_interp["gx"].values
    df["gy"] = gyro_interp["gy"].values
    df["gz"] = gyro_interp["gz"].values

    return df.reset_index(drop=True)



def segment_reps(df, window_seconds=1.5, step_seconds=1.0):
    """
    Fallback: split the whole recording into overlapping fixed windows.
    Assumes roughly 1 rep per ~1–2 seconds.
    """
    dt = float(df["time"].iloc[1] - df["time"].iloc[0])
    win_samples = int(window_seconds / dt)
    step_samples = int(step_seconds / dt)

    rep_windows = []
    n = len(df)
    start = 0
    while start + win_samples <= n:
        end = start + win_samples
        rep_windows.append((start, end))
        start += step_samples

    print(f"Fixed-window segmentation: built {len(rep_windows)} windows "
          f"(window_seconds={window_seconds}, step_seconds={step_seconds})")
    return rep_windows





def extract_features_from_window(df, start, end):
    window = df.iloc[start:end]

    features = []

    for col in ["ax", "ay", "az", "gx", "gy", "gz"]:
        vals = window[col].values
        features.append(np.mean(vals))
        features.append(np.std(vals))
        features.append(np.min(vals))
        features.append(np.max(vals))
        features.append(np.max(vals) - np.min(vals))

    ax = window["ax"].values
    ay = window["ay"].values
    az = window["az"].values
    accel_mag = np.sqrt(ax ** 2 + ay ** 2 + az ** 2)
    features.append(np.max(accel_mag))
    features.append(np.std(np.diff(accel_mag)))

    duration = float(window["time"].iloc[-1] - window["time"].iloc[0])
    features.append(duration)

    return np.array(features, dtype=float)


def main():
    if len(sys.argv) != 2:
        print("Usage: python classify_new_session.py <session_folder>")
        print("Example: python classify_new_session.py data/correct")
        sys.exit(1)

    session_folder = sys.argv[1]
    acc_path = os.path.join(session_folder, ACC_FILE)
    gyro_path = os.path.join(session_folder, GYRO_FILE)

    if not (os.path.isfile(acc_path) and os.path.isfile(gyro_path)):
        print(f"Could not find {ACC_FILE} and {GYRO_FILE} in {session_folder}")
        sys.exit(1)

    model_path = os.path.join("models", "bicep_model.joblib")
    if not os.path.isfile(model_path):
        print(f"Model not found at {model_path}. Run train_bicep_model.py first.")
        sys.exit(1)

    print(f"Loading data from: {session_folder}")
    df = load_and_merge(acc_path, gyro_path)

    windows = segment_reps(df)
    if not windows:
        print("No reps detected.")
        sys.exit(0)

    X = []
    for (s, e) in windows:
        feats = extract_features_from_window(df, s, e)
        X.append(feats)

    X = np.vstack(X)

    clf = joblib.load(model_path)
    preds = clf.predict(X)
    probs = clf.predict_proba(X)

    for i, (label, p) in enumerate(zip(preds, probs), start=1):
        cls = "CORRECT" if label == 0 else "INCORRECT"
        confidence = float(np.max(p))
        print(f"Rep {i:02d}: {cls} (confidence {confidence:.2f})")


if __name__ == "__main__":
    main()
