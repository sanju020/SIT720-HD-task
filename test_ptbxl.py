import pandas as pd
import wfdb

DATASET_DIR = r"C:\Users\sanju\Downloads\ptb-xl"

df = pd.read_csv(
    DATASET_DIR + r"\ptbxl_database.csv",
    index_col="ecg_id"
)

print("Metadata shape:", df.shape)

first_record = df.iloc[0]["filename_lr"]

print("First waveform path:", first_record)

record_path = DATASET_DIR + "\\" + first_record.replace("/", "\\")

signal, fields = wfdb.rdsamp(record_path)

print("\nSignal shape:", signal.shape)
print("Sampling frequency:", fields["fs"])
print("Lead names:", fields["sig_name"])
print("\nFirst 5 samples:")
print(signal[:5])