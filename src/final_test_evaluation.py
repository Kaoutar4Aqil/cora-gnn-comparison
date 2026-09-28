
import csv
from pathlib import Path

import numpy as np
import torch

from sklearn.metrics import accuracy_score, f1_score
from torch_geometric.datasets import Planetoid

from gcn_model import GCNModel
from gatv2_model import GATv2Model
from appnp_model import APPNPModel


# ============================================================
# 1. CONFIGURATION
# ============================================================

SEEDS = [7, 21, 42, 123, 2024]
MODELS = ["GCN", "GATv2", "APPNP"]

DEVICE = torch.device("cpu")

METRICS_DIR = Path("outputs/metrics")
METRICS_DIR.mkdir(parents=True, exist_ok=True)

DETAIL_PATH = METRICS_DIR / "final_test_all_seeds.csv"
SUMMARY_PATH = METRICS_DIR / "final_test_summary.csv"


# ============================================================
# 2. DATASET
# ============================================================

dataset = Planetoid(
    root="data/Planetoid",
    name="Cora",
    split="public",
)

data = dataset[0].to(DEVICE)

assert int(data.train_mask.sum()) == 140
assert int(data.val_mask.sum()) == 500
assert int(data.test_mask.sum()) == 1000


# ============================================================
# 3. MODEL FACTORY
# ============================================================

def build_model(name, checkpoint):

    config = {
        "in_channels": checkpoint["in_channels"],
        "hidden_channels": checkpoint["hidden_channels"],
        "out_channels": checkpoint["out_channels"],
        "dropout": checkpoint["dropout"],
    }

    if name == "GCN":
        return GCNModel(**config)

    if name == "GATv2":
        return GATv2Model(
            **config,
            heads=8,
        )

    if name == "APPNP":
        return APPNPModel(
            **config,
            propagation_steps=10,
            alpha=0.1,
        )

    raise ValueError(f"Unknown model: {name}")


# ============================================================
# 4. CHECK ALL 15 CHECKPOINTS EXIST FIRST
# ============================================================

checkpoints = []

for name in MODELS:
    for seed in SEEDS:

        path = Path(
            f"models/{name.lower()}_best_seed{seed}.pt"
        )

        if not path.is_file():
            raise FileNotFoundError(
                f"Missing checkpoint: {path}"
            )

        checkpoints.append((name, seed, path))

print("=" * 65)
print("CORA - FINAL TEST EVALUATION")
print("=" * 65)

print(f"\nDevice: {DEVICE}")
print(f"Test nodes: {int(data.test_mask.sum())}")
print(f"Checkpoints found: {len(checkpoints)}")
print("Model selection: validation loss")
print("Evaluation: accuracy and macro-F1")

assert len(checkpoints) == 15


# ============================================================
# 5. EVALUATE EACH SAVED CHECKPOINT
# ============================================================

results = []

for name, seed, path in checkpoints:

    checkpoint = torch.load(
        path,
        map_location=DEVICE,
        weights_only=True,
    )

    assert checkpoint["model_name"] == name
    assert int(checkpoint["seed"]) == seed

    model = build_model(
        name,
        checkpoint,
    ).to(DEVICE)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    with torch.no_grad():

        logits = model(
            data.x,
            data.edge_index,
        )

        predictions = logits.argmax(dim=1)

    y_true = (
        data.y[data.test_mask]
        .cpu()
        .numpy()
    )

    y_pred = (
        predictions[data.test_mask]
        .cpu()
        .numpy()
    )

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        labels=list(range(dataset.num_classes)),
        zero_division=0,
    )

    result = {
        "model": name,
        "seed": seed,
        "best_epoch": int(checkpoint["epoch"]),
        "test_accuracy": float(accuracy),
        "test_macro_f1": float(macro_f1),
    }

    results.append(result)

    print(
        f"{name:6s} | "
        f"Seed {seed:4d} | "
        f"Accuracy: {accuracy:.4f} | "
        f"Macro-F1: {macro_f1:.4f}",
        flush=True,
    )


# ============================================================
# 6. SAVE RESULTS PER SEED
# ============================================================

with DETAIL_PATH.open(
    "w",
    newline="",
    encoding="utf-8",
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "model",
            "seed",
            "best_epoch",
            "test_accuracy",
            "test_macro_f1",
        ],
    )

    writer.writeheader()
    writer.writerows(results)


# ============================================================
# 7. MULTI-SEED SUMMARY
# ============================================================

summary = []

print("\n" + "=" * 65)
print("FINAL TEST SUMMARY - MEAN +/- SAMPLE STD")
print("=" * 65)

for name in MODELS:

    model_results = [
        row for row in results
        if row["model"] == name
    ]

    accuracies = np.asarray([
        row["test_accuracy"]
        for row in model_results
    ])

    macro_f1_values = np.asarray([
        row["test_macro_f1"]
        for row in model_results
    ])

    accuracy_mean = float(accuracies.mean())
    accuracy_std = float(accuracies.std(ddof=1))

    f1_mean = float(macro_f1_values.mean())
    f1_std = float(macro_f1_values.std(ddof=1))

    summary.append({
        "model": name,
        "num_seeds": len(model_results),
        "accuracy_mean": accuracy_mean,
        "accuracy_std": accuracy_std,
        "macro_f1_mean": f1_mean,
        "macro_f1_std": f1_std,
    })

    print(
        f"{name:6s} | "
        f"Accuracy: {accuracy_mean:.4f} "
        f"+/- {accuracy_std:.4f} | "
        f"Macro-F1: {f1_mean:.4f} "
        f"+/- {f1_std:.4f}"
    )


with SUMMARY_PATH.open(
    "w",
    newline="",
    encoding="utf-8",
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "model",
            "num_seeds",
            "accuracy_mean",
            "accuracy_std",
            "macro_f1_mean",
            "macro_f1_std",
        ],
    )

    writer.writeheader()
    writer.writerows(summary)


print("\nALL TEST EVALUATIONS COMPLETED")

print(f"\nPer-seed results: {DETAIL_PATH}")
print(f"Summary:          {SUMMARY_PATH}")
print("\nDo not tune models using these test results.")
