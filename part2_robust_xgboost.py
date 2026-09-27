from pathlib import Path
import pandas as pd

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

import matplotlib.pyplot as plt


DATASET_DIR = Path(
    r"C:\Users\sanju\Downloads\ptb-xl"
)

INPUT_FILE = (
    DATASET_DIR /
    "part2_robust_features.csv"
)

RANDOM_STATE = 42


# ---------------------------------------------------------
# Load
# ---------------------------------------------------------

df = pd.read_csv(INPUT_FILE)

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

y = df["diagnostic_class"].map(mapping)


print(
    "Dataset shape:",
    X.shape
)

print(
    "\nClass mapping:",
    mapping
)


# ---------------------------------------------------------
# Same split as Part 1
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = (
    train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y
    )
)


# ---------------------------------------------------------
# Proposed pipeline
# ---------------------------------------------------------

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
    "\nTraining proposed robust-feature XGBoost..."
)

pipeline.fit(
    X_train,
    y_train
)


pred = pipeline.predict(
    X_test
)

prob = pipeline.predict_proba(
    X_test
)


# ---------------------------------------------------------
# Evaluation
# ---------------------------------------------------------

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
print("ROBUST FEATURE PART 2 RESULTS")
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
            for i in range(len(labels))
        ],
        zero_division=0
    )
)


# ---------------------------------------------------------
# Comparison
# ---------------------------------------------------------

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
            "Robust_Feature_XGBoost":
                proposed[metric],
            "Difference":
                proposed[metric]
                - baseline[metric]
        }
    )

comparison = pd.DataFrame(rows)

print("\n================================")
print("BASELINE VS ROBUST FEATURES")
print("================================")

print(
    comparison.to_string(
        index=False
    )
)

comparison.to_csv(
    DATASET_DIR /
    "part2_robust_vs_baseline.csv",
    index=False
)


# ---------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------

cm = confusion_matrix(
    y_test,
    pred
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        reverse_mapping[i]
        for i in range(len(labels))
    ]
)

disp.plot(
    xticks_rotation=45
)

plt.title(
    "Part 2 Robust Feature XGBoost"
)

plt.tight_layout()

plt.savefig(
    DATASET_DIR /
    "part2_robust_confusion_matrix.png",
    dpi=300
)

plt.close()