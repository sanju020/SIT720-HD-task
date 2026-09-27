from pathlib import Path
import numpy as np
import pandas as pd
import wfdb
import neurokit2 as nk
import time

# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

DATASET_DIR = Path(r"C:\Users\sanju\Downloads\ptb-xl")

INPUT_FILE = DATASET_DIR / "ptbxl_single_label.csv"

CHECKPOINT_FILE = DATASET_DIR / "dasmmc_beat_features_checkpoint.csv"

FAILED_FILE = DATASET_DIR / "dasmmc_failed_ecgs.csv"

FINAL_FILE = DATASET_DIR / "dasmmc_beat_features_full.csv"

CHECKPOINT_EVERY = 100


# ---------------------------------------------------------
# LOAD CLEAN SINGLE-LABEL DATA
# ---------------------------------------------------------

df = pd.read_csv(INPUT_FILE)

print("Single-label ECG records:", len(df))
print("\nClass distribution:")
print(df["diagnostic_class"].value_counts())


# ---------------------------------------------------------
# SUPPORT FUNCTIONS
# ---------------------------------------------------------

def clean_peak_array(values):

    if values is None:
        return np.array([], dtype=int)

    values = np.asarray(values, dtype=float)

    values = values[np.isfinite(values)]

    return values.astype(int)


def nearest_before(values, r_peak):

    candidates = values[values < r_peak]

    if len(candidates) == 0:
        return np.nan

    return candidates[-1]


def nearest_after(values, r_peak):

    candidates = values[values > r_peak]

    if len(candidates) == 0:
        return np.nan

    return candidates[0]


# ---------------------------------------------------------
# FEATURE EXTRACTION
# ---------------------------------------------------------

def extract_beats(record_path):

    signal, fields = wfdb.rdsamp(str(record_path))

    fs = int(fields["fs"])

    lead_names = fields["sig_name"]

    if "II" not in lead_names:
        raise ValueError("Lead II unavailable.")

    lead_index = lead_names.index("II")

    ecg = signal[:, lead_index]

    # ECG cleaning
    cleaned = nk.ecg_clean(
        ecg,
        sampling_rate=fs,
        method="neurokit"
    )

    # R-peak detection
    _, rpeaks = nk.ecg_peaks(
        cleaned,
        sampling_rate=fs
    )

    r = clean_peak_array(
        rpeaks["ECG_R_Peaks"]
    )

    if len(r) < 2:
        raise ValueError("Less than two R peaks detected.")

    # PQRST delineation
    _, waves = nk.ecg_delineate(
        cleaned,
        r,
        sampling_rate=fs,
        method="dwt"
    )

    p_all = clean_peak_array(
        waves.get("ECG_P_Peaks")
    )

    q_all = clean_peak_array(
        waves.get("ECG_Q_Peaks")
    )

    s_all = clean_peak_array(
        waves.get("ECG_S_Peaks")
    )

    t_all = clean_peak_array(
        waves.get("ECG_T_Peaks")
    )

    beats = []

    for i, r_peak in enumerate(r):

        p = nearest_before(p_all, r_peak)
        q = nearest_before(q_all, r_peak)

        s = nearest_after(s_all, r_peak)
        t = nearest_after(t_all, r_peak)

        # Skip incomplete beats
        if any(
            np.isnan(x)
            for x in [p, q, s, t]
        ):
            continue

        p = int(p)
        q = int(q)
        s = int(s)
        t = int(t)

        # PQRST order validation
        if not (
            p < q < r_peak < s < t
        ):
            continue

        # RR interval
        if i < len(r) - 1:

            rr = (
                r[i + 1] - r_peak
            ) / fs

        else:

            rr = np.nan

        beat = {

            # -------------------------
            # Peak locations
            # -------------------------

            "P_location": p / fs,
            "Q_location": q / fs,
            "R_location": r_peak / fs,
            "S_location": s / fs,
            "T_location": t / fs,

            # -------------------------
            # Peak amplitudes
            # -------------------------

            "P_amplitude": cleaned[p],
            "Q_amplitude": cleaned[q],
            "R_amplitude": cleaned[r_peak],
            "S_amplitude": cleaned[s],
            "T_amplitude": cleaned[t],

            # -------------------------
            # ECG intervals
            # -------------------------

            "PQ_interval": (q - p) / fs,

            "ST_interval": (t - s) / fs,

            "QT_interval": (t - q) / fs,

            "PR_interval": (
                r_peak - p
            ) / fs,

            "RR_interval": rr,

            "QRS_interval": (
                s - q
            ) / fs
        }

        beats.append(beat)

    if len(beats) == 0:
        raise ValueError(
            "No complete valid beats found."
        )

    return beats


# ---------------------------------------------------------
# RESUME EXISTING CHECKPOINT
# ---------------------------------------------------------

if CHECKPOINT_FILE.exists():

    print("\nCheckpoint detected.")

    existing = pd.read_csv(
        CHECKPOINT_FILE
    )

    completed_ids = set(
        existing["ecg_id"].unique()
    )

    results = existing.to_dict(
        orient="records"
    )

    print(
        "Previously completed ECGs:",
        len(completed_ids)
    )

else:

    completed_ids = set()

    results = []


# ---------------------------------------------------------
# FAILED ECG CHECKPOINT
# ---------------------------------------------------------

if FAILED_FILE.exists():

    failed_df = pd.read_csv(
        FAILED_FILE
    )

    failed_ids = set(
        failed_df["ecg_id"]
    )

    failed_records = (
        failed_df.to_dict(
            orient="records"
        )
    )

else:

    failed_ids = set()

    failed_records = []


# ---------------------------------------------------------
# DETERMINE RECORDS STILL TO PROCESS
# ---------------------------------------------------------

skip_ids = completed_ids.union(
    failed_ids
)

remaining_df = df[
    ~df["ecg_id"].isin(skip_ids)
].copy()

print(
    "\nRemaining ECGs:",
    len(remaining_df)
)


# ---------------------------------------------------------
# FULL EXTRACTION
# ---------------------------------------------------------

start_time = time.time()

processed_since_save = 0

for position, (_, row) in enumerate(
    remaining_df.iterrows(),
    start=1
):

    ecg_id = row["ecg_id"]

    label = row["diagnostic_class"]

    record_path = (
        DATASET_DIR /
        row["filename_lr"]
    )

    print(
        f"[{position}/{len(remaining_df)}] "
        f"ECG {ecg_id} | {label}",
        end=""
    )

    try:

        beats = extract_beats(
            record_path
        )

        for beat_number, beat in enumerate(
            beats
        ):

            result = {

                "ecg_id": ecg_id,

                "diagnostic_class":
                    label,

                "beat_number":
                    beat_number
            }

            result.update(beat)

            results.append(result)

        completed_ids.add(ecg_id)

        print(
            f" -> {len(beats)} beats"
        )

    except Exception as e:

        print(
            f" -> FAILED: {e}"
        )

        failed_records.append(
            {
                "ecg_id": ecg_id,
                "diagnostic_class":
                    label,
                "filename_lr":
                    row["filename_lr"],
                "error": str(e)
            }
        )

        failed_ids.add(ecg_id)

    processed_since_save += 1


    # -------------------------------------------------
    # CHECKPOINT
    # -------------------------------------------------

    if (
        processed_since_save
        >= CHECKPOINT_EVERY
    ):

        pd.DataFrame(
            results
        ).to_csv(
            CHECKPOINT_FILE,
            index=False
        )

        pd.DataFrame(
            failed_records
        ).to_csv(
            FAILED_FILE,
            index=False
        )

        processed_since_save = 0

        elapsed = (
            time.time() -
            start_time
        )

        print(
            "\n--- CHECKPOINT SAVED ---"
        )

        print(
            "Completed ECGs:",
            len(completed_ids)
        )

        print(
            "Failed ECGs:",
            len(failed_ids)
        )

        print(
            "Beat rows:",
            len(results)
        )

        print(
            "Elapsed minutes:",
            round(
                elapsed / 60,
                2
            )
        )

        print(
            "------------------------\n"
        )


# ---------------------------------------------------------
# FINAL SAVE
# ---------------------------------------------------------

feature_df = pd.DataFrame(
    results
)

feature_df.to_csv(
    FINAL_FILE,
    index=False
)

pd.DataFrame(
    failed_records
).to_csv(
    FAILED_FILE,
    index=False
)


# ---------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------

elapsed = (
    time.time() -
    start_time
)

print("\n================================")
print("FEATURE EXTRACTION COMPLETE")
print("================================")

print(
    "ECGs successfully processed:",
    feature_df["ecg_id"].nunique()
)

print(
    "Failed ECGs:",
    len(failed_ids)
)

print(
    "Total beat rows:",
    len(feature_df)
)

print(
    "Total time (minutes):",
    round(
        elapsed / 60,
        2
    )
)

print("\nBeat count per ECG:")

beat_counts = (
    feature_df
    .groupby("ecg_id")
    .size()
)

print(
    beat_counts.describe()
)

print("\nMissing values:")

print(
    feature_df.isna().sum()
)

print("\nFinal file:")
print(FINAL_FILE)