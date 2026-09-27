from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

DATASET_DIR = Path(__file__).resolve().parent

BASELINE_FILE = DATASET_DIR / "baseline_results.csv"
EFFICIENCY_FILE = DATASET_DIR / "part2_efficiency_comparison.csv"


# ---------------------------------------------------------
# LOAD RESULTS
# ---------------------------------------------------------

baseline = pd.read_csv(BASELINE_FILE)
efficiency = pd.read_csv(EFFICIENCY_FILE)

print("Baseline results:")
print(baseline)

print("\nEfficiency results:")
print(efficiency)


# =========================================================
# FIGURE 1
# PUBLISHED VS REPRODUCED ACCURACY
# =========================================================

published_accuracy = {
    "XGBoost": 93.0,
    "Random Forest": 91.0,
    "CatBoost": 89.0,
    "Gradient Boosting": 88.0,
    "KNN": 82.0
}

model_order = [
    "XGBoost",
    "Random Forest",
    "CatBoost",
    "Gradient Boosting",
    "KNN"
]

reproduced_accuracy = []

for model in model_order:

    row = baseline[
        baseline["Model"] == model
    ]

    if len(row) == 0:
        raise ValueError(
            f"{model} was not found in baseline_results.csv"
        )

    reproduced_accuracy.append(
        row.iloc[0]["Accuracy"] * 100
    )


published_values = [
    published_accuracy[m]
    for m in model_order
]

x = np.arange(
    len(model_order)
)

width = 0.36

fig, ax = plt.subplots(
    figsize=(10, 6)
)

ax.bar(
    x - width / 2,
    published_values,
    width,
    label="Published DASMcC"
)

ax.bar(
    x + width / 2,
    reproduced_accuracy,
    width,
    label="Reproduced"
)

ax.set_ylabel(
    "Accuracy (%)"
)

ax.set_xlabel(
    "Classifier"
)

ax.set_title(
    "Published vs Reproduced Classifier Accuracy"
)

ax.set_xticks(x)

ax.set_xticklabels(
    model_order,
    rotation=20,
    ha="right"
)

ax.set_ylim(
    0,
    100
)

ax.legend()

ax.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

figure1 = (
    DATASET_DIR /
    "published_vs_reproduced_accuracy.png"
)

plt.savefig(
    figure1,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# =========================================================
# FIGURE 2
# BASELINE VS ROBUST PREDICTIVE PERFORMANCE
# =========================================================

baseline_row = efficiency[
    efficiency["Representation"]
    == "Baseline Sequential Features"
].iloc[0]

robust_row = efficiency[
    efficiency["Representation"]
    == "Robust Statistical Features"
].iloc[0]


metrics = [
    "Accuracy",
    "AUC",
    "Macro_F1"
]

metric_labels = [
    "Accuracy",
    "AUC",
    "Macro-F1"
]

baseline_values = [
    baseline_row[m]
    for m in metrics
]

robust_values = [
    robust_row[m]
    for m in metrics
]


x = np.arange(
    len(metrics)
)

fig, ax = plt.subplots(
    figsize=(8, 6)
)

ax.bar(
    x - width / 2,
    baseline_values,
    width,
    label="Baseline sequential"
)

ax.bar(
    x + width / 2,
    robust_values,
    width,
    label="Robust statistical"
)

ax.set_ylabel(
    "Score"
)

ax.set_title(
    "Baseline vs Proposed Predictive Performance"
)

ax.set_xticks(x)

ax.set_xticklabels(
    metric_labels
)

ax.set_ylim(
    0,
    1
)

ax.legend()

ax.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

figure2 = (
    DATASET_DIR /
    "part2_predictive_comparison.png"
)

plt.savefig(
    figure2,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# =========================================================
# FIGURE 3
# PART 2 EFFICIENCY REDUCTIONS
# =========================================================

feature_reduction = (
    (
        baseline_row["Number_of_Features"]
        -
        robust_row["Number_of_Features"]
    )
    /
    baseline_row["Number_of_Features"]
) * 100


training_reduction = (
    (
        baseline_row["Training_Time_Seconds"]
        -
        robust_row["Training_Time_Seconds"]
    )
    /
    baseline_row["Training_Time_Seconds"]
) * 100


prediction_reduction = (
    (
        baseline_row["Prediction_Time_Seconds"]
        -
        robust_row["Prediction_Time_Seconds"]
    )
    /
    baseline_row["Prediction_Time_Seconds"]
) * 100


model_size_reduction = (
    (
        baseline_row["Model_Size_MB"]
        -
        robust_row["Model_Size_MB"]
    )
    /
    baseline_row["Model_Size_MB"]
) * 100


categories = [
    "Features",
    "Training time",
    "Prediction time",
    "Model size"
]

reductions = [
    feature_reduction,
    training_reduction,
    prediction_reduction,
    model_size_reduction
]


fig, ax = plt.subplots(
    figsize=(9, 6)
)

bars = ax.bar(
    categories,
    reductions
)

ax.set_ylabel(
    "Reduction (%)"
)

ax.set_title(
    "Efficiency Improvements of Robust Representation"
)

ax.set_ylim(
    0,
    max(reductions) + 10
)

ax.grid(
    axis="y",
    alpha=0.3
)


for bar, value in zip(
    bars,
    reductions
):

    ax.text(
        bar.get_x()
        + bar.get_width() / 2,

        bar.get_height()
        + 1,

        f"{value:.2f}%",

        ha="center",
        va="bottom"
    )


plt.tight_layout()

figure3 = (
    DATASET_DIR /
    "part2_efficiency_reductions.png"
)

plt.savefig(
    figure3,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ---------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------

print("\n========================================")
print("REPORT FIGURES GENERATED")
print("========================================")

print("Figure 1:")
print(figure1)

print("\nFigure 2:")
print(figure2)

print("\nFigure 3:")
print(figure3)

print("\nCalculated reductions:")

print(
    f"Feature reduction: "
    f"{feature_reduction:.2f}%"
)

print(
    f"Training time reduction: "
    f"{training_reduction:.2f}%"
)

print(
    f"Prediction time reduction: "
    f"{prediction_reduction:.2f}%"
)

print(
    f"Model size reduction: "
    f"{model_size_reduction:.2f}%"
)