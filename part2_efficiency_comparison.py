from pathlib import Path
import time
import os
import joblib

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

from xgboost import XGBClassifier


# =========================================================
# PATHS / SETTINGS
# =========================================================

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

RESULT_FILE = (
    DATASET_DIR /
    "part2_efficiency_comparison.csv"
)

RANDOM_STATE = 42


# =========================================================
# COMMON MODEL
# =========================================================

def make_pipeline():

    return Pipeline(
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
                    n_estimators=200,
                    learning_rate=0.05,
                    max_depth=6,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective="multi:softprob",
                    eval_metric="mlogloss",
                    random_state=RANDOM_STATE,
                    n_jobs=-1
                )
            )
        ]
    )


# =========================================================
# EVALUATION FUNCTION
# =========================================================

def evaluate_dataset(
    df,
    representation_name,
    model_filename
):

    X = df.drop(
        columns=[
            "ecg_id",
            "diagnostic_class"
        ]
    )

    classes = sorted(
        df["diagnostic_class"].unique()
    )

    mapping = {
        label: i
        for i, label in enumerate(classes)
    }

    y = (
        df["diagnostic_class"]
        .map(mapping)
    )


    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=RANDOM_STATE,
            stratify=y
        )
    )


    pipeline = make_pipeline()


    print(
        f"\nTraining {representation_name}..."
    )


    # ---------------------------------------------
    # TRAINING TIME
    # ---------------------------------------------

    start_fit = time.perf_counter()

    pipeline.fit(
        X_train,
        y_train
    )

    fit_time = (
        time.perf_counter()
        - start_fit
    )


    # ---------------------------------------------
    # PREDICTION TIME
    # ---------------------------------------------

    start_pred = time.perf_counter()

    pred = pipeline.predict(
        X_test
    )

    prob = pipeline.predict_proba(
        X_test
    )

    prediction_time = (
        time.perf_counter()
        - start_pred
    )


    # ---------------------------------------------
    # METRICS
    # ---------------------------------------------

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


    # ---------------------------------------------
    # SAVE MODEL AND GET FILE SIZE
    # ---------------------------------------------

    model_path = (
        DATASET_DIR /
        model_filename
    )

    joblib.dump(
        pipeline,
        model_path
    )

    model_size_mb = (
        os.path.getsize(
            model_path
        )
        /
        (1024 * 1024)
    )


    return {
        "Representation":
            representation_name,

        "Number_of_Features":
            X.shape[1],

        "Accuracy":
            accuracy,

        "Precision":
            precision,

        "Recall":
            recall,

        "Weighted_F1":
            weighted_f1,

        "Macro_F1":
            macro_f1,

        "AUC":
            auc,

        "Training_Time_Seconds":
            fit_time,

        "Prediction_Time_Seconds":
            prediction_time,

        "Model_Size_MB":
            model_size_mb
    }


# =========================================================
# LOAD DATA
# =========================================================

baseline_df = pd.read_csv(
    BASELINE_FILE
)

robust_df = pd.read_csv(
    ROBUST_FILE
)


# =========================================================
# RUN CONTROLLED COMPARISON
# =========================================================

baseline_results = evaluate_dataset(
    baseline_df,
    "Baseline Sequential Features",
    "baseline_efficiency_model.joblib"
)

robust_results = evaluate_dataset(
    robust_df,
    "Robust Statistical Features",
    "robust_efficiency_model.joblib"
)


results_df = pd.DataFrame(
    [
        baseline_results,
        robust_results
    ]
)


# =========================================================
# DISPLAY
# =========================================================

print("\n========================================")
print("CONTROLLED EFFICIENCY COMPARISON")
print("========================================")

print(
    results_df.to_string(
        index=False
    )
)


# =========================================================
# IMPROVEMENT CALCULATIONS
# =========================================================

feature_reduction = (
    (
        baseline_results[
            "Number_of_Features"
        ]
        -
        robust_results[
            "Number_of_Features"
        ]
    )
    /
    baseline_results[
        "Number_of_Features"
    ]
) * 100


training_change = (
    (
        baseline_results[
            "Training_Time_Seconds"
        ]
        -
        robust_results[
            "Training_Time_Seconds"
        ]
    )
    /
    baseline_results[
        "Training_Time_Seconds"
    ]
) * 100


prediction_change = (
    (
        baseline_results[
            "Prediction_Time_Seconds"
        ]
        -
        robust_results[
            "Prediction_Time_Seconds"
        ]
    )
    /
    baseline_results[
        "Prediction_Time_Seconds"
    ]
) * 100


print("\n========================================")
print("EFFICIENCY SUMMARY")
print("========================================")

print(
    f"Feature reduction: "
    f"{feature_reduction:.2f}%"
)

print(
    f"Training time reduction: "
    f"{training_change:.2f}%"
)

print(
    f"Prediction time reduction: "
    f"{prediction_change:.2f}%"
)

print(
    f"AUC difference: "
    f"{robust_results['AUC'] - baseline_results['AUC']:.6f}"
)

print(
    f"Accuracy difference: "
    f"{robust_results['Accuracy'] - baseline_results['Accuracy']:.6f}"
)


# =========================================================
# SAVE
# =========================================================

results_df.to_csv(
    RESULT_FILE,
    index=False
)

print(
    "\nSaved comparison to:"
)

print(
    RESULT_FILE
)