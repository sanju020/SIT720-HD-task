from pathlib import Path
import warnings

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression

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

warnings.filterwarnings("ignore")


# =========================================================
# SETTINGS
# =========================================================

DATASET_DIR = Path(
    r"C:\Users\sanju\Downloads\ptb-xl"
)

INPUT_FILE = (
    DATASET_DIR /
    "dasmmc_model_data.csv"
)

RESULT_FILE = (
    DATASET_DIR /
    "part2_stacking_results.csv"
)

CLASS_RESULT_FILE = (
    DATASET_DIR /
    "part2_classification_report.csv"
)

RANDOM_STATE = 42


# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_csv(INPUT_FILE)

X = df.drop(
    columns=[
        "ecg_id",
        "diagnostic_class"
    ]
)

y_text = df[
    "diagnostic_class"
]

classes = sorted(
    y_text.unique()
)

mapping = {
    label: i
    for i, label in enumerate(classes)
}

reverse_mapping = {
    i: label
    for label, i in mapping.items()
}

y = y_text.map(mapping)


print(
    "Dataset shape:",
    X.shape
)

print(
    "\nClass mapping:"
)

print(mapping)


# =========================================================
# SAME 80/20 SPLIT AS BASELINE
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

print(
    "\nTraining samples:",
    len(X_train)
)

print(
    "Testing samples:",
    len(X_test)
)


# =========================================================
# BASE MODELS
# =========================================================

rf = RandomForestClassifier(
    n_estimators=150,
    max_depth=None,
    random_state=RANDOM_STATE,
    n_jobs=-1
)


xgb = XGBClassifier(
    n_estimators=150,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="multi:softprob",
    eval_metric="mlogloss",
    random_state=RANDOM_STATE,
    n_jobs=-1
)


# =========================================================
# META-LEARNER
# =========================================================

meta_model = LogisticRegression(
    max_iter=3000,
    random_state=RANDOM_STATE
)


# =========================================================
# STACKING MODEL
# =========================================================

stacking = StackingClassifier(
    estimators=[
        ("rf", rf),
        ("xgb", xgb)
    ],

    final_estimator=meta_model,

    stack_method="predict_proba",

    cv=5,

    n_jobs=-1,

    passthrough=False
)


# =========================================================
# LEAKAGE-SAFE PIPELINE
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
            "stacking",
            stacking
        )
    ]
)


# =========================================================
# TRAIN
# =========================================================

print(
    "\n=================================="
)

print(
    "Training Part 2 Stacking Ensemble"
)

print(
    "=================================="
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

f1_weighted = f1_score(
    y_test,
    pred,
    average="weighted",
    zero_division=0
)

f1_macro = f1_score(
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


# =========================================================
# DISPLAY RESULTS
# =========================================================

print(
    "\n=================================="
)

print(
    "PART 2 RESULTS"
)

print(
    "=================================="
)

print(
    f"Accuracy    : {accuracy:.4f}"
)

print(
    f"Precision   : {precision:.4f}"
)

print(
    f"Recall      : {recall:.4f}"
)

print(
    f"Weighted F1 : {f1_weighted:.4f}"
)

print(
    f"Macro F1    : {f1_macro:.4f}"
)

print(
    f"AUC         : {auc:.4f}"
)


# =========================================================
# CLASSIFICATION REPORT
# =========================================================

target_names = [
    reverse_mapping[i]
    for i in range(
        len(classes)
    )
]

print(
    "\nClassification report:\n"
)

print(
    classification_report(
        y_test,
        pred,
        target_names=target_names,
        zero_division=0
    )
)


report_dict = classification_report(
    y_test,
    pred,
    target_names=target_names,
    zero_division=0,
    output_dict=True
)

report_df = pd.DataFrame(
    report_dict
).transpose()

report_df.to_csv(
    CLASS_RESULT_FILE
)


# =========================================================
# SAVE SUMMARY
# =========================================================

result_df = pd.DataFrame(
    [
        {
            "Model":
                "RF + XGBoost Stacking",

            "Accuracy":
                accuracy,

            "Precision":
                precision,

            "Recall":
                recall,

            "Weighted_F1":
                f1_weighted,

            "Macro_F1":
                f1_macro,

            "AUC":
                auc
        }
    ]
)

result_df.to_csv(
    RESULT_FILE,
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
    display_labels=target_names
)

disp.plot(
    xticks_rotation=45
)

plt.title(
    "Part 2 - Stacking Ensemble Confusion Matrix"
)

plt.tight_layout()

CM_FILE = (
    DATASET_DIR /
    "part2_stacking_confusion_matrix.png"
)

plt.savefig(
    CM_FILE,
    dpi=300
)

plt.close()


# =========================================================
# BASELINE COMPARISON
# =========================================================

BASELINE_XGB = {
    "Accuracy": 0.696351,
    "Precision": 0.675045,
    "Recall": 0.696351,
    "F1": 0.681375,
    "AUC": 0.845467
}

comparison = pd.DataFrame(
    [
        {
            "Metric": "Accuracy",
            "Baseline_XGBoost":
                BASELINE_XGB["Accuracy"],
            "Proposed_Stacking":
                accuracy
        },

        {
            "Metric": "Precision",
            "Baseline_XGBoost":
                BASELINE_XGB["Precision"],
            "Proposed_Stacking":
                precision
        },

        {
            "Metric": "Recall",
            "Baseline_XGBoost":
                BASELINE_XGB["Recall"],
            "Proposed_Stacking":
                recall
        },

        {
            "Metric": "F1",
            "Baseline_XGBoost":
                BASELINE_XGB["F1"],
            "Proposed_Stacking":
                f1_weighted
        },

        {
            "Metric": "AUC",
            "Baseline_XGBoost":
                BASELINE_XGB["AUC"],
            "Proposed_Stacking":
                auc
        }
    ]
)

comparison[
    "Difference"
] = (
    comparison["Proposed_Stacking"]
    -
    comparison["Baseline_XGBoost"]
)

COMPARISON_FILE = (
    DATASET_DIR /
    "part2_vs_baseline.csv"
)

comparison.to_csv(
    COMPARISON_FILE,
    index=False
)

print(
    "\n=================================="
)

print(
    "BASELINE VS PROPOSED"
)

print(
    "=================================="
)

print(
    comparison.to_string(
        index=False
    )
)

print(
    "\nSaved:"
)

print(
    RESULT_FILE
)

print(
    CLASS_RESULT_FILE
)

print(
    COMPARISON_FILE
)

print(
    CM_FILE
)