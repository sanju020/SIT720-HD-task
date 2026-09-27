from pathlib import Path
import numpy as np
import pandas as pd
import wfdb
import neurokit2 as nk
import warnings
import time

warnings.filterwarnings("ignore")

DATASET_DIR = Path(r"C:\Users\sanju\Downloads\ptb-xl")

INPUT_FILE = DATASET_DIR / "ptbxl_single_label.csv"
OUTPUT_FILE = DATASET_DIR / "dasmmc_12lead_pilot.csv"

RANDOM_STATE = 42
FIXED_BEATS = 11
PILOT_SIZE = 1500

FEATURE_NAMES = [
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


def clean_peaks(values):
    if values is None:
        return np.array([], dtype=int)

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    return values.astype(int)


def nearest_before(values, r):
    x = values[values < r]
    return np.nan if len(x) == 0 else x[-1]


def nearest_after(values, r):
    x = values[values > r]
    return np.nan if len(x) == 0 else x[0]


def extract_lead(signal, fs):

    cleaned = nk.ecg_clean(
        signal,
        sampling_rate=fs,
        method="neurokit"
    )

    _, peaks = nk.ecg_peaks(
        cleaned,
        sampling_rate=fs
    )

    r = clean_peaks(
        peaks["ECG_R_Peaks"]
    )

    if len(r) < 2:
        return None

    _, waves = nk.ecg_delineate(
        cleaned,
        r,
        sampling_rate=fs,
        method="dwt"
    )

    p_all = clean_peaks(
        waves.get("ECG_P_Peaks")
    )
    q_all = clean_peaks(
        waves.get("ECG_Q_Peaks")
    )
    s_all = clean_peaks(
        waves.get("ECG_S_Peaks")
    )
    t_all = clean_peaks(
        waves.get("ECG_T_Peaks")
    )

    beats = []

    for i, r_peak in enumerate(r):

        p = nearest_before(p_all, r_peak)
        q = nearest_before(q_all, r_peak)
        s = nearest_after(s_all, r_peak)
        t = nearest_after(t_all, r_peak)

        if any(
            np.isnan(x)
            for x in [p, q, s, t]
        ):
            continue

        p, q, s, t = map(
            int,
            [p, q, s, t]
        )

        if not (
            p < q < r_peak < s < t
        ):
            continue

        if i < len(r) - 1:
            rr = (
                r[i + 1] - r_peak
            ) / fs
        else:
            rr = np.nan

        beats.append([
            p / fs,
            q / fs,
            r_peak / fs,
            s / fs,
            t / fs,

            cleaned[p],
            cleaned[q],
            cleaned[r_peak],
            cleaned[s],
            cleaned[t],

            (q - p) / fs,
            (t - s) / fs,
            (t - q) / fs,
            (r_peak - p) / fs,
            rr,
            (s - q) / fs
        ])

    if len(beats) == 0:
        return None

    beats = np.asarray(
        beats,
        dtype=float
    )

    # Fill missing RR values
    beat_df = pd.DataFrame(
        beats,
        columns=FEATURE_NAMES
    )

    beat_df["RR_interval"] = (
        beat_df["RR_interval"]
        .ffill()
        .bfill()
    )

    beats = beat_df.to_numpy()

    # Equalise to 11 beats
    if len(beats) >= FIXED_BEATS:

        beats = beats[:FIXED_BEATS]

    else:

        padding = np.repeat(
            beats[-1:],
            FIXED_BEATS - len(beats),
            axis=0
        )

        beats = np.vstack([
            beats,
            padding
        ])

    return beats.flatten()


# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------

df = pd.read_csv(INPUT_FILE)

# stratified pilot sample
pilot_parts = []

for label, group in df.groupby(
    "diagnostic_class"
):

    proportion = (
        len(group) / len(df)
    )

    n = max(
        1,
        round(
            PILOT_SIZE * proportion
        )
    )

    sample = group.sample(
        n=min(n, len(group)),
        random_state=RANDOM_STATE
    )

    pilot_parts.append(sample)

pilot_df = pd.concat(
    pilot_parts
).sample(
    frac=1,
    random_state=RANDOM_STATE
).reset_index(drop=True)

print(
    "Pilot ECGs:",
    len(pilot_df)
)

print(
    pilot_df[
        "diagnostic_class"
    ].value_counts()
)

results = []

start = time.time()

for pos, (_, row) in enumerate(
    pilot_df.iterrows(),
    start=1
):

    record_path = (
        DATASET_DIR /
        row["filename_lr"]
    )

    try:

        signal, fields = wfdb.rdsamp(
            str(record_path)
        )

        fs = int(fields["fs"])

        lead_vectors = []

        for lead_idx in range(
            signal.shape[1]
        ):

            try:

                vec = extract_lead(
                    signal[:, lead_idx],
                    fs
                )

                if vec is not None:
                    lead_vectors.append(
                        vec
                    )

            except Exception:
                pass

        if len(lead_vectors) == 0:
            raise ValueError(
                "No usable leads."
            )

        lead_vectors = np.vstack(
            lead_vectors
        )

        # Median feature representation
        # across all successfully processed leads
        final_vector = np.nanmedian(
            lead_vectors,
            axis=0
        )

        result = {
            "ecg_id": row["ecg_id"],
            "diagnostic_class":
                row["diagnostic_class"],
            "usable_leads":
                len(lead_vectors)
        }

        for i, value in enumerate(
            final_vector
        ):
            result[
                f"f_{i:03d}"
            ] = value

        results.append(result)

        print(
            f"[{pos}/{len(pilot_df)}] "
            f"ECG {row['ecg_id']} "
            f"-> {len(lead_vectors)} leads"
        )

    except Exception as e:

        print(
            f"[{pos}/{len(pilot_df)}] "
            f"FAILED ECG {row['ecg_id']}: "
            f"{e}"
        )


result_df = pd.DataFrame(
    results
)

result_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n==========================")
print("12-LEAD PILOT COMPLETE")
print("==========================")

print(
    "Successful ECGs:",
    len(result_df)
)

print(
    "\nClass distribution:"
)

print(
    result_df[
        "diagnostic_class"
    ].value_counts()
)

print(
    "\nUsable leads:"
)

print(
    result_df[
        "usable_leads"
    ].describe()
)

print(
    "\nMissing feature values:",
    result_df.isna().sum().sum()
)

print(
    "\nElapsed minutes:",
    round(
        (time.time() - start) / 60,
        2
    )
)

print(
    "\nSaved:",
    OUTPUT_FILE
)