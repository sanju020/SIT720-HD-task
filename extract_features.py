from pathlib import Path
import numpy as np
import pandas as pd
import wfdb
import neurokit2 as nk

DATASET_DIR = Path(r"C:\Users\sanju\Downloads\ptb-xl")
LABEL_FILE = DATASET_DIR / "ptbxl_single_label.csv"

df = pd.read_csv(LABEL_FILE)

# Test 20 records first
df = df.head(20).copy()


def clean_peak_array(values):
    """Convert NeuroKit peak output into valid integer indices."""
    if values is None:
        return np.array([], dtype=int)

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    return values.astype(int)


def nearest_before(values, r):
    """Nearest peak occurring before R."""
    candidates = values[values < r]

    if len(candidates) == 0:
        return np.nan

    return candidates[-1]


def nearest_after(values, r):
    """Nearest peak occurring after R."""
    candidates = values[values > r]

    if len(candidates) == 0:
        return np.nan

    return candidates[0]


def extract_beats(record_path):

    signal, fields = wfdb.rdsamp(str(record_path))

    sampling_rate = int(fields["fs"])
    lead_names = fields["sig_name"]

    # Assumption: Lead II used for morphology extraction
    lead_index = lead_names.index("II")
    ecg = signal[:, lead_index]

    cleaned = nk.ecg_clean(
        ecg,
        sampling_rate=sampling_rate,
        method="neurokit"
    )

    # Detect R peaks
    _, rpeaks = nk.ecg_peaks(
        cleaned,
        sampling_rate=sampling_rate
    )

    r = clean_peak_array(rpeaks["ECG_R_Peaks"])

    if len(r) < 2:
        raise ValueError("Too few R peaks detected.")

    # ECG delineation
    _, waves = nk.ecg_delineate(
        cleaned,
        r,
        sampling_rate=sampling_rate,
        method="dwt"
    )

    p_all = clean_peak_array(waves.get("ECG_P_Peaks"))
    q_all = clean_peak_array(waves.get("ECG_Q_Peaks"))
    s_all = clean_peak_array(waves.get("ECG_S_Peaks"))
    t_all = clean_peak_array(waves.get("ECG_T_Peaks"))

    beats = []

    for i, r_peak in enumerate(r):

        p = nearest_before(p_all, r_peak)
        q = nearest_before(q_all, r_peak)

        s = nearest_after(s_all, r_peak)
        t = nearest_after(t_all, r_peak)

        # Skip beat if complete PQRST sequence is unavailable
        if any(np.isnan(x) for x in [p, q, s, t]):
            continue

        p = int(p)
        q = int(q)
        s = int(s)
        t = int(t)

        # Require correct physiological ordering
        if not (p < q < r_peak < s < t):
            continue

        # RR interval uses next R peak
        if i < len(r) - 1:
            rr = (r[i + 1] - r_peak) / sampling_rate
        else:
            rr = np.nan

        beat = {
            # Peak locations (seconds)
            "P_location": p / sampling_rate,
            "Q_location": q / sampling_rate,
            "R_location": r_peak / sampling_rate,
            "S_location": s / sampling_rate,
            "T_location": t / sampling_rate,

            # Amplitudes
            "P_amplitude": cleaned[p],
            "Q_amplitude": cleaned[q],
            "R_amplitude": cleaned[r_peak],
            "S_amplitude": cleaned[s],
            "T_amplitude": cleaned[t],

            # Intervals
            "PQ_interval": (q - p) / sampling_rate,
            "ST_interval": (t - s) / sampling_rate,
            "QT_interval": (t - q) / sampling_rate,
            "PR_interval": (r_peak - p) / sampling_rate,
            "RR_interval": rr,
            "QRS_interval": (s - q) / sampling_rate
        }

        beats.append(beat)

    return beats


all_results = []

for count, (_, row) in enumerate(df.iterrows(), start=1):

    record_path = DATASET_DIR / row["filename_lr"]

    print(
        f"Processing ECG {row['ecg_id']} "
        f"({count}/{len(df)})"
    )

    try:

        beats = extract_beats(record_path)

        print(f"  Valid beats: {len(beats)}")

        for beat_number, beat in enumerate(beats):

            result = {
                "ecg_id": row["ecg_id"],
                "diagnostic_class": row["diagnostic_class"],
                "beat_number": beat_number
            }

            result.update(beat)

            all_results.append(result)

    except Exception as e:

        print(
            f"  Could not process ECG "
            f"{row['ecg_id']}: {e}"
        )


beat_df = pd.DataFrame(all_results)

print("\nFeature dataset shape:")
print(beat_df.shape)

print("\nPreview:")
print(beat_df.head(20))

print("\nInterval minimums:")
for col in [
    "PQ_interval",
    "ST_interval",
    "QT_interval",
    "PR_interval",
    "RR_interval",
    "QRS_interval"
]:
    print(
        col,
        beat_df[col].min()
    )

print("\nInterval maximums:")
for col in [
    "PQ_interval",
    "ST_interval",
    "QT_interval",
    "PR_interval",
    "RR_interval",
    "QRS_interval"
]:
    print(
        col,
        beat_df[col].max()
    )

output_path = (
    DATASET_DIR /
    "dasmmc_beat_features_test.csv"
)

beat_df.to_csv(
    output_path,
    index=False
)

print("\nSaved to:")
print(output_path)