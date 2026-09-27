from pathlib import Path
import numpy as np
import pandas as pd

DATASET_DIR = Path(r"C:\Users\sanju\Downloads\ptb-xl")

INPUT_FILE = DATASET_DIR / "dasmmc_beat_features_full.csv"
OUTPUT_FILE = DATASET_DIR / "part2_robust_features.csv"

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

print("Beat rows:", len(df))
print("Unique ECGs:", df["ecg_id"].nunique())


# ---------------------------------------------------------
# Handle final-beat missing RR values
# ---------------------------------------------------------

df["RR_interval"] = (
    df.groupby("ecg_id")["RR_interval"]
      .transform(lambda x: x.ffill().bfill())
)


# ---------------------------------------------------------
# Build robust ECG-level features
# ---------------------------------------------------------

records = []

for ecg_id, group in df.groupby("ecg_id"):

    label = group["diagnostic_class"].iloc[0]

    result = {
        "ecg_id": ecg_id,
        "diagnostic_class": label,
        "valid_beat_count": len(group)
    }

    for feature in FEATURES:

        values = group[feature].dropna().to_numpy()

        if len(values) == 0:
            continue

        result[f"{feature}_mean"] = np.mean(values)

        result[f"{feature}_median"] = np.median(values)

        result[f"{feature}_std"] = (
            np.std(values)
            if len(values) > 1
            else 0.0
        )

        result[f"{feature}_min"] = np.min(values)

        result[f"{feature}_max"] = np.max(values)

        q75, q25 = np.percentile(
            values,
            [75, 25]
        )

        result[f"{feature}_iqr"] = q75 - q25

    # Additional physiologically interpretable feature
    mean_rr = group["RR_interval"].mean()

    if mean_rr > 0:
        result["mean_heart_rate"] = 60.0 / mean_rr
    else:
        result["mean_heart_rate"] = np.nan

    records.append(result)


feature_df = pd.DataFrame(records)


# ---------------------------------------------------------
# Missing values
# ---------------------------------------------------------

numeric_cols = feature_df.select_dtypes(
    include=[np.number]
).columns

for col in numeric_cols:

    if col == "ecg_id":
        continue

    feature_df[col] = (
        feature_df[col]
        .fillna(
            feature_df[col].median()
        )
    )


print("\nPart 2 dataset shape:")
print(feature_df.shape)

print("\nClass distribution:")
print(
    feature_df[
        "diagnostic_class"
    ].value_counts()
)

print(
    "\nMissing values:",
    feature_df.isna().sum().sum()
)

feature_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nSaved:")
print(OUTPUT_FILE)