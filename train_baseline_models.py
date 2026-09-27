from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.pipeline import Pipeline as SkPipeline

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

from xgboost import XGBClassifier
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")

DATASET_DIR = Path(r"C:\Users\sanju\Downloads\ptb-xl")

INPUT_FILE = DATASET_DIR / "dasmmc_model_data.csv"
RESULTS_FILE = DATASET_DIR / "baseline_results.csv"

RANDOM_STATE = 42


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

df = pd.read_csv(INPUT_FILE)

X = df.drop(
    columns=[
        "ecg_id",
        "diagnostic_class"
    ]
)

y = df["diagnostic_class"]

print("Dataset shape:", X.shape)

print("\nClass distribution:")
print(y.value_counts())


# ---------------------------------------------------------
# LABEL ENCODING
# ---------------------------------------------------------

class_names = sorted(y.unique())

class_to_int = {
    label: idx
    for idx, label in enumerate(class_names)
}

int_to_class = {
    value: key
    for key, value in class_to_int.items()
}

y_encoded = y.map(class_to_int)

print("\nClass mapping:")
print(class_to_int)


# ---------------------------------------------------------
# 80/20 STRATIFIED SPLIT
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y_encoded
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

print("\nTraining class distribution:")
print(y_train.value_counts().sort_index())


# ---------------------------------------------------------
# MODELS
# ---------------------------------------------------------

models = {

    "XGBoost": XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),

    "CatBoost": CatBoostClassifier(
        iterations=200,
        learning_rate=0.05,
        depth=6,
        random_seed=RANDOM_STATE,
        verbose=False,
        thread_count=-1
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=50,
        learning_rate=0.1,
        max_depth=2,
        random_state=RANDOM_STATE
    )
}

# ---------------------------------------------------------
# TRAIN + TEST
# ---------------------------------------------------------

results = []

trained_models = {}

for name, model in models.items():

    print("\n====================================")
    print("Training:", name)
    print("====================================")

    # Scaling is especially useful for KNN
    # SMOTE is applied only to training data
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "smote",
                SMOTE(
                    random_state=RANDOM_STATE
                )
            ),
            ("model", model)
        ]
    )

    pipeline.fit(
        X_train,
        y_train
    )

    trained_models[name] = pipeline

    y_pred = pipeline.predict(
        X_test
    )

    y_prob = pipeline.predict_proba(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    auc = roc_auc_score(
        y_test,
        y_prob,
        multi_class="ovr",
        average="weighted"
    )

    results.append(
        {
            "Model": name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "AUC": auc
        }
    )

    print(
        f"Accuracy : {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall   : {recall:.4f}"
    )

    print(
        f"F1 Score : {f1:.4f}"
    )

    print(
        f"AUC      : {auc:.4f}"
    )

    print("\nClassification report:")

    print(
        classification_report(
            y_test,
            y_pred,
            target_names=[
                int_to_class[i]
                for i in range(
                    len(class_names)
                )
            ],
            zero_division=0
        )
    )


# ---------------------------------------------------------
# RESULTS TABLE
# ---------------------------------------------------------

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="Accuracy",
    ascending=False
)

print("\n====================================")
print("FINAL BASELINE RESULTS")
print("====================================")

print(results_df)

results_df.to_csv(
    RESULTS_FILE,
    index=False
)

print("\nSaved results to:")
print(RESULTS_FILE)


# ---------------------------------------------------------
# BEST MODEL CONFUSION MATRIX
# ---------------------------------------------------------

best_model_name = results_df.iloc[0]["Model"]

best_model = trained_models[
    best_model_name
]

best_pred = best_model.predict(
    X_test
)

cm = confusion_matrix(
    y_test,
    best_pred
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        int_to_class[i]
        for i in range(
            len(class_names)
        )
    ]
)

disp.plot(
    xticks_rotation=45
)

plt.title(
    f"Confusion Matrix - {best_model_name}"
)

plt.tight_layout()

cm_path = DATASET_DIR / "baseline_confusion_matrix.png"

plt.savefig(
    cm_path,
    dpi=300
)

plt.close()

print("\nSaved confusion matrix:")
print(cm_path)


# ---------------------------------------------------------
# 10-FOLD CROSS-VALIDATION
# ---------------------------------------------------------

print("\n====================================")
print("10-FOLD CROSS-VALIDATION")
print("====================================")

cv = StratifiedKFold(
    n_splits=10,
    shuffle=True,
    random_state=RANDOM_STATE
)

cv_results = []

for name, model in models.items():

    print("\nCross-validating:", name)

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "smote",
                SMOTE(
                    random_state=RANDOM_STATE
                )
            ),
            ("model", model)
        ]
    )

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision_weighted",
        "recall": "recall_weighted",
        "f1": "f1_weighted",
        "auc": "roc_auc_ovr_weighted"
    }

    scores = cross_validate(
        pipeline,
        X,
        y_encoded,
        cv=cv,
        scoring=scoring,
        n_jobs=1
    )

    cv_row = {
        "Model": name,
        "CV Accuracy":
            scores[
                "test_accuracy"
            ].mean(),

        "CV Precision":
            scores[
                "test_precision"
            ].mean(),

        "CV Recall":
            scores[
                "test_recall"
            ].mean(),

        "CV F1":
            scores[
                "test_f1"
            ].mean(),

        "CV AUC":
            scores[
                "test_auc"
            ].mean()
    }

    cv_results.append(
        cv_row
    )

    print(cv_row)


cv_df = pd.DataFrame(
    cv_results
)

cv_path = (
    DATASET_DIR /
    "baseline_cv_results.csv"
)

cv_df.to_csv(
    cv_path,
    index=False
)

print("\nSaved CV results:")
print(cv_path)