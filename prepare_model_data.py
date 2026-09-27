from pathlib import Path
import numpy as np
import pandas as pd

DATASET_DIR = Path(r"C:\Users\sanju\Downloads\ptb-xl")

INPUT_FILE = DATASET_DIR / "dasmmc_beat_features_full.csv"
OUTPUT_FILE = DATASET_DIR / "dasmmc_model_data.csv"

FIXED_BEATS = 11

FEATURES = [
    "P_location",
    "Q_location",
    "R_location",
    "S_location",
    "T_location",
    "P_amplitude",
    "Q_amplitude",
    "R_amplitude",
    "S_amplitude",
    "T_amplitude",
    "PQ_interval",
    "ST_interval",
    "QT_interval",
    "PR_interval",
    "RR_interval",
    "QRS_interval"
]

df = pd.read_csv(INPUT_FILE)

print("Input beat rows:", len(df))
print("Unique ECGs:", df["ecg_id"].nunique())

# Fill RR gaps within each ECG
df["RR_interval"] = (
    df.groupby("ecg_id")["RR_interval"]
      .transform(lambda s: s.ffill().bfill())
)

records = []

for ecg_id, group in df.groupby("ecg_id"):

    group = group.sort_values("beat_number").copy()

    label = group["diagnostic_class"].iloc[0]

    values = group[FEATURES].to_numpy()

    # truncate
    if len(values) >= FIXED_BEATS:
        values = values[:FIXED_BEATS]

    # pad by repeating last valid beat
    else:
        pad_count = FIXED_BEATS - len(values)
        last_row = values[-1:]

        padding = np.repeat(
            last_row,
            pad_count,
            axis=0
        )

        values = np.vstack([
            values,
            padding
        ])

    flat = values.flatten()

    record = {
        "ecg_id": ecg_id,
        "diagnostic_class": label
    }

    for i, value in enumerate(flat):
        record[f"f_{i:03d}"] = value

    records.append(record)

model_df = pd.DataFrame(records)

print("\nModel dataset shape:")
print(model_df.shape)

print("\nClass distribution:")
print(model_df["diagnostic_class"].value_counts())

print("\nMissing values:")
print(model_df.isna().sum().sum())

model_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nSaved to:")
print(OUTPUT_FILE)