from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

from xgboost import XGBClassifier


DATASET_DIR = Path(
    r"C:\Users\sanju\Downloads\ptb-xl"
)

BASELINE_FILE = (
    DATASET_DIR /
    "dasmmc_model_data.csv"
)

ROBUST_FILE = (
    DATASET_DIR /
    "part2_robust_features.csv"
)

RANDOM_STATE = 42


# =========================================================
# LOAD BOTH REPRESENTATIONS
# =========================================================

baseline_df = pd.read_csv(
    BASELINE_FILE
)

robust_df = pd.read_csv(
    ROBUST_FILE
)


# Rename robust features to prevent column-name collisions
robust_feature_cols = [
    c for c in robust_df.columns
    if c not in [
        "ecg_id",
        "diagnostic_class"
    ]
]

robust_df = robust_df.rename(
    columns={
        c: f"robust_{c}"
        for c in robust_feature_cols
    }
)


# =========================================================
# MERGE BY ECG ID
# =========================================================

df = baseline_df.merge(
    robust_df.drop(
        columns=["diagnostic_class"]
    ),
    on="ecg_id",
    how="inner"
)

print(
    "Fused dataset shape:",
    df.shape
)

print(
    "\nECGs:",
    df["ecg_id"].nunique()
)


# =========================================================
# INPUT / LABELS
# =========================================================

X = df.drop(
    columns=[
        "ecg_id",
        "diagnostic_class"
    ]
)

labels = sorted(
    df["diagnostic_class"].unique()
)

mapping = {
    label: i
    for i, label in enumerate(labels)
}

reverse_mapping = {
    value: key
    for key, value in mapping.items()
}

y = (
    df["diagnostic_class"]
    .map(mapping)
)


print(
    "\nNumber of fused features:",
    X.shape[1]
)


# =========================================================
# SAME SPLIT AS BASELINE
# =========================================================

X_train, X_test, y_train, y_test = (
    train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y
    )
)


# =========================================================
# PIPELINE
# =========================================================

pipeline = Pipeline(
    steps=[
        (
            "scaler",
            StandardScaler()
        ),

        (
            "smote",
            SMOTE(
                random_state=RANDOM_STATE
            )
        ),

        (
            "model",
            XGBClassifier(
                n_estimators=300,
                learning_rate=0.04,
                max_depth=6,
                min_child_weight=2,
                subsample=0.85,
                colsample_bytree=0.85,
                objective="multi:softprob",
                eval_metric="mlogloss",
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
        )
    ]
)


print(
    "\nTraining fused-feature XGBoost..."
)

pipeline.fit(
    X_train,
    y_train
)


# =========================================================
# PREDICTIONS
# =========================================================

pred = pipeline.predict(
    X_test
)

prob = pipeline.predict_proba(
    X_test
)


# =========================================================
# METRICS
# =========================================================

accuracy = accuracy_score(
    y_test,
    pred
)

precision = precision_score(
    y_test,
    pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_test,
    pred,
    average="weighted",
    zero_division=0
)

weighted_f1 = f1_score(
    y_test,
    pred,
    average="weighted",
    zero_division=0
)

macro_f1 = f1_score(
    y_test,
    pred,
    average="macro",
    zero_division=0
)

auc = roc_auc_score(
    y_test,
    prob,
    multi_class="ovr",
    average="weighted"
)


print("\n================================")
print("FUSED FEATURE PART 2 RESULTS")
print("================================")

print(f"Accuracy    : {accuracy:.4f}")
print(f"Precision   : {precision:.4f}")
print(f"Recall      : {recall:.4f}")
print(f"Weighted F1 : {weighted_f1:.4f}")
print(f"Macro F1    : {macro_f1:.4f}")
print(f"AUC         : {auc:.4f}")


print("\nClassification report:\n")

print(
    classification_report(
        y_test,
        pred,
        target_names=[
            reverse_mapping[i]
            for i in range(
                len(labels)
            )
        ],
        zero_division=0
    )
)


# =========================================================
# BASELINE COMPARISON
# =========================================================

baseline = {
    "Accuracy": 0.696351,
    "Precision": 0.675045,
    "Recall": 0.696351,
    "Weighted F1": 0.681375,
    "AUC": 0.845467
}

proposed = {
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "Weighted F1": weighted_f1,
    "AUC": auc
}

rows = []

for metric in baseline:

    rows.append(
        {
            "Metric": metric,
            "Baseline_XGBoost":
                baseline[metric],

            "Fused_XGBoost":
                proposed[metric],

            "Difference":
                proposed[metric]
                - baseline[metric]
        }
    )

comparison = pd.DataFrame(
    rows
)

print("\n================================")
print("BASELINE VS FUSED FEATURES")
print("================================")

print(
    comparison.to_string(
        index=False
    )
)

comparison.to_csv(
    DATASET_DIR /
    "part2_fused_vs_baseline.csv",
    index=False
)


# =========================================================
# CONFUSION MATRIX
# =========================================================

cm = confusion_matrix(
    y_test,
    pred
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        reverse_mapping[i]
        for i in range(
            len(labels)
        )
    ]
)

disp.plot(
    xticks_rotation=45
)

plt.title(
    "Part 2 - Fused Feature XGBoost"
)

plt.tight_layout()

plt.savefig(
    DATASET_DIR /
    "part2_fused_confusion_matrix.png",
    dpi=300
)

plt.close()