import os
import time
import numpy as np
import pandas as pd
import requests

# Use one of your existing sessions as a "live" source
ACC_FILE = os.path.join("data", "correct", "Accelerometer.csv")
GYRO_FILE = os.path.join("data", "correct", "Gyroscope.csv")

# If you want to test heavy swing instead, change to "incorrect"
# ACC_FILE = os.path.join("data", "incorrect", "Accelerometer.csv")
# GYRO_FILE = os.path.join("data", "incorrect", "Gyroscope.csv")

# FastAPI server URL
API_URL = "http://127.0.0.1:8000/classify_window"


def load_and_merge_for_sim(acc_path, gyro_path, resample_dt=0.01):
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

    # Drop duplicate timestamps
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


def main():
    df = load_and_merge_for_sim(ACC_FILE, GYRO_FILE)
    print(f"Loaded {len(df)} samples")

    # Match your training segmentation
    window_seconds = 1.5
    step_seconds = 1.0

    dt = float(df["time"].iloc[1] - df["time"].iloc[0])
    win_samples = int(window_seconds / dt)
    step_samples = int(step_seconds / dt)

    start = 0
    window_idx = 1

    while start + win_samples <= len(df):
        end = start + win_samples
        window = df.iloc[start:end]

        payload = {
            "samples": [
                {
                    "time": float(row["time"]),
                    "ax": float(row["ax"]),
                    "ay": float(row["ay"]),
                    "az": float(row["az"]),
                    "gx": float(row["gx"]),
                    "gy": float(row["gy"]),
                    "gz": float(row["gz"]),
                }
                for _, row in window.iterrows()
            ]
        }

        try:
            r = requests.post(API_URL, json=payload, timeout=5)
            r.raise_for_status()
            resp = r.json()
            print(f"Window {window_idx:02d}: {resp}")
        except Exception as e:
            print(f"Window {window_idx:02d}: ERROR {e}")

        window_idx += 1
        start += step_samples

        # Sleep to simulate real time
        time.sleep(step_seconds)

    print("Done simulation.")


if __name__ == "__main__":
    main()
