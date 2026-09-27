import ast
from pathlib import Path

import pandas as pd

DATASET_DIR = Path(r"C:\Users\sanju\Downloads\ptb-xl")

# --------------------------------------------------
# LOAD FILES
# --------------------------------------------------

df = pd.read_csv(DATASET_DIR / "ptbxl_database.csv")

scp = pd.read_csv(
    DATASET_DIR / "scp_statements.csv",
    index_col=0
)

print("PTB-XL shape:", df.shape)
print("SCP statements shape:", scp.shape)

# --------------------------------------------------
# PARSE SCP CODES
# --------------------------------------------------

df["scp_codes"] = df["scp_codes"].apply(ast.literal_eval)

# Keep only diagnostic SCP statements
diagnostic_scp = scp[scp["diagnostic"] == 1]

print("\nDiagnostic SCP statements:")
print(diagnostic_scp["diagnostic_class"].value_counts(dropna=False))

# --------------------------------------------------
# MAP EACH ECG TO DIAGNOSTIC SUPERCLASSES
# --------------------------------------------------

def get_superclasses(code_dict):
    classes = set()

    for code in code_dict.keys():
        if code in diagnostic_scp.index:
            diagnostic_class = diagnostic_scp.loc[code, "diagnostic_class"]

            if pd.notna(diagnostic_class):
                classes.add(diagnostic_class)

    return sorted(classes)


df["diagnostic_superclasses"] = df["scp_codes"].apply(get_superclasses)

print("\nExamples:")
print(
    df[
        ["ecg_id", "scp_codes", "diagnostic_superclasses"]
    ].head(10)
)

# --------------------------------------------------
# COUNT NUMBER OF SUPERCLASSES
# --------------------------------------------------

df["num_superclasses"] = df["diagnostic_superclasses"].apply(len)

print("\nNumber of diagnostic superclasses per ECG:")
print(df["num_superclasses"].value_counts().sort_index())

# --------------------------------------------------
# KEEP SINGLE-LABEL RECORDS ONLY
# --------------------------------------------------

single_label_df = df[df["num_superclasses"] == 1].copy()

single_label_df["diagnostic_class"] = (
    single_label_df["diagnostic_superclasses"]
    .apply(lambda x: x[0])
)

# Keep only the five classes used in DASMcC
VALID_CLASSES = ["NORM", "MI", "STTC", "HYP", "CD"]

single_label_df = single_label_df[
    single_label_df["diagnostic_class"].isin(VALID_CLASSES)
].copy()

print("\nFinal single-label dataset shape:")
print(single_label_df.shape)

print("\nClass distribution:")
print(single_label_df["diagnostic_class"].value_counts())

# --------------------------------------------------
# SAVE CLEAN LABEL DATASET
# --------------------------------------------------

output_path = DATASET_DIR / "ptbxl_single_label.csv"

single_label_df.to_csv(output_path, index=False)

print("\nSaved to:")
print(output_path)