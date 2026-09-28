
# ============================================================
# CORA - GCN VS GATv2 VS APPNP
# Training curves and final test visualizations
# ============================================================

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# 1. CONFIGURATION
# ============================================================

OUTPUT_DIR = Path("outputs/figures")
METRICS_DIR = Path("outputs/metrics")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MODELS = {
    "GCN": "gcn_training_seed42.csv",
    "GATv2": "gatv2_training_seed42.csv",
    "APPNP": "appnp_training_seed42.csv",
}


# ============================================================
# 2. FIGURE 1 - TRAINING CURVES
# ============================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(14, 5),
)

for model_name, filename in MODELS.items():

    history_path = METRICS_DIR / filename

    if not history_path.is_file():
        raise FileNotFoundError(
            f"Missing training history: {history_path}"
        )

    history = pd.read_csv(history_path)

    # Validation loss
    axes[0].plot(
        history["epoch"],
        history["val_loss"],
        label=model_name,
        linewidth=2,
    )

    # Validation accuracy
    axes[1].plot(
        history["epoch"],
        history["val_accuracy"] * 100,
        label=model_name,
        linewidth=2,
    )

axes[0].set_title(
    "Validation Loss - Seed 42"
)

axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Cross-Entropy Loss")

axes[1].set_title(
    "Validation Accuracy - Seed 42"
)

axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy (%)")

for ax in axes:
    ax.grid(alpha=0.25)
    ax.legend()

fig.suptitle(
    "Cora - GCN vs GATv2 vs APPNP",
    fontsize=15,
    fontweight="bold",
)

fig.tight_layout()

training_path = (
    OUTPUT_DIR / "model_training_comparison.png"
)

fig.savefig(
    training_path,
    dpi=200,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# 3. FIGURE 2 - FINAL TEST PERFORMANCE
# ============================================================

summary_path = (
    METRICS_DIR / "final_test_summary.csv"
)

if not summary_path.is_file():
    raise FileNotFoundError(
        f"Missing final test results: {summary_path}"
    )

summary = pd.read_csv(summary_path)

summary = summary.set_index("model").loc[
    ["GCN", "GATv2", "APPNP"]
]

names = summary.index.tolist()

# Convert fractions into percentages.
accuracy = (
    summary["accuracy_mean"].to_numpy() * 100
)

accuracy_std = (
    summary["accuracy_std"].to_numpy() * 100
)

macro_f1 = (
    summary["macro_f1_mean"].to_numpy() * 100
)

macro_f1_std = (
    summary["macro_f1_std"].to_numpy() * 100
)

positions = np.arange(len(names))
width = 0.35

fig, ax = plt.subplots(
    figsize=(10, 6)
)

# Mean accuracy with standard deviation.
bars_accuracy = ax.bar(
    positions - width / 2,
    accuracy,
    width,
    yerr=accuracy_std,
    capsize=5,
    label="Accuracy",
)

# Mean macro-F1 with standard deviation.
bars_f1 = ax.bar(
    positions + width / 2,
    macro_f1,
    width,
    yerr=macro_f1_std,
    capsize=5,
    label="Macro-F1",
)

ax.set_xticks(positions)
ax.set_xticklabels(names)

ax.set_ylim(0, 100)
ax.set_ylabel("Score (%)")

ax.set_title(
    "Cora - Final Test Performance Across 5 Seeds",
    fontsize=14,
    fontweight="bold",
)

ax.legend()
ax.grid(axis="y", alpha=0.2)
ax.set_axisbelow(True)

# Show accuracy values above the bars.
ax.bar_label(
    bars_accuracy,
    fmt="%.2f",
    padding=7,
)

# Show macro-F1 values above the bars.
ax.bar_label(
    bars_f1,
    fmt="%.2f",
    padding=7,
)

fig.tight_layout()

test_path = (
    OUTPUT_DIR / "final_test_model_comparison.png"
)

fig.savefig(
    test_path,
    dpi=200,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# 4. RESULTS
# ============================================================

print("=" * 60)
print("CORA - MODEL COMPARISON VISUALIZATIONS")
print("=" * 60)

print("\nGenerated figures:")
print(training_path)
print(test_path)

print("\nAll plots generated successfully!")
print("Test results used for reporting only.")
