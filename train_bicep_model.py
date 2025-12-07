import os
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib


DATA_DIR = "data"
CORRECT_DIR = os.path.join(DATA_DIR, "correct")
INCORRECT_DIR = os.path.join(DATA_DIR, "incorrect")

ACC_FILE = "Accelerometer.csv"
GYRO_FILE = "Gyroscope.csv"


def load_and_merge(acc_path, gyro_path, resample_dt=0.01):
    """
    Load accelerometer and gyroscope CSVs and merge on a regular time grid.
    Handles duplicate timestamps by keeping the first occurrence.
    """
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

    # Drop duplicate timestamps (keep first) before using as index
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
    """
    Extract simple statistical features from a rep window.
    """
    window = df.iloc[start:end]

    features = []

    for col in ["ax", "ay", "az", "gx", "gy", "gz"]:
        vals = window[col].values
        features.append(np.mean(vals))
        features.append(np.std(vals))
        features.append(np.min(vals))
        features.append(np.max(vals))
        features.append(np.max(vals) - np.min(vals))

    # Acceleration magnitude smoothness etc.
    ax = window["ax"].values
    ay = window["ay"].values
    az = window["az"].values
    accel_mag = np.sqrt(ax ** 2 + ay ** 2 + az ** 2)
    features.append(np.max(accel_mag))
    features.append(np.std(np.diff(accel_mag)))  # rough smoothness measure

    # Duration of rep
    duration = float(window["time"].iloc[-1] - window["time"].iloc[0])
    features.append(duration)

    return np.array(features, dtype=float)


def build_dataset():
    X_list = []
    y_list = []

    # CORRECT
    correct_acc = os.path.join(CORRECT_DIR, ACC_FILE)
    correct_gyro = os.path.join(CORRECT_DIR, GYRO_FILE)

    correct_df = load_and_merge(correct_acc, correct_gyro)
    print(f"Correct df length: {len(correct_df)}")
    correct_windows = segment_reps(correct_df)
    print(f"Correct reps: {len(correct_windows)}")

    for (s, e) in correct_windows:
        feats = extract_features_from_window(correct_df, s, e)
        X_list.append(feats)
        y_list.append(0)  # correct

    # INCORRECT
    incorrect_acc = os.path.join(INCORRECT_DIR, ACC_FILE)
    incorrect_gyro = os.path.join(INCORRECT_DIR, GYRO_FILE)

    incorrect_df = load_and_merge(incorrect_acc, incorrect_gyro)
    print(f"Incorrect df length: {len(incorrect_df)}")
    incorrect_windows = segment_reps(incorrect_df)
    print(f"Incorrect reps: {len(incorrect_windows)}")

    for (s, e) in incorrect_windows:
        feats = extract_features_from_window(incorrect_df, s, e)
        X_list.append(feats)
        y_list.append(1)  # incorrect

    if not X_list:
        raise RuntimeError(
            "No rep windows were detected. Try lowering prominence or distance_seconds in segment_reps()."
        )

    X = np.vstack(X_list)
    y = np.array(y_list, dtype=int)

    return X, y



def main():
    print("Building dataset from CSVs...")
    X, y = build_dataset()
    print(f"Total reps: {len(y)}, features per rep: {X.shape[1]}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )

    print("Training RandomForest model...")
    clf.fit(X_train, y_train)

    print("Evaluation on held-out test reps:")
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred, digits=3))

    # Save model
    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", "bicep_model.joblib")
    joblib.dump(clf, model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()
