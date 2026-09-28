
import csv
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from torch_geometric.datasets import Planetoid

from gcn_model import GCNModel
from gatv2_model import GATv2Model
from appnp_model import APPNPModel


# ============================================================
# CONFIGURATION
# ============================================================

SEEDS = [7, 21, 123, 2024]
MODELS = ["GCN", "GATv2", "APPNP"]

HIDDEN = 64
DROPOUT = 0.5
LR = 0.01
WEIGHT_DECAY = 5e-4

MAX_EPOCHS = 300
PATIENCE = 50

DEVICE = torch.device("cpu")

OUTPUT = Path(
    "outputs/metrics/multiseed_validation_summary.csv"
)

Path("models").mkdir(exist_ok=True)
OUTPUT.parent.mkdir(parents=True, exist_ok=True)


# ============================================================
# DATASET
# ============================================================

dataset = Planetoid(
    root="data/Planetoid",
    name="Cora",
    split="public",
)

data = dataset[0].to(DEVICE)

assert int(data.train_mask.sum()) == 140
assert int(data.val_mask.sum()) == 500


# ============================================================
# MODEL FACTORY
# ============================================================

def create_model(name):

    kwargs = {
        "in_channels": dataset.num_features,
        "hidden_channels": HIDDEN,
        "out_channels": dataset.num_classes,
        "dropout": DROPOUT,
    }

    if name == "GCN":
        return GCNModel(**kwargs)

    if name == "GATv2":
        return GATv2Model(**kwargs, heads=8)

    if name == "APPNP":
        return APPNPModel(
            **kwargs,
            propagation_steps=10,
            alpha=0.1,
        )

    raise ValueError(f"Unknown model: {name}")


def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ============================================================
# VALIDATION - NO TEST METRICS
# ============================================================

@torch.no_grad()
def validate(model):

    model.eval()

    logits = model(data.x, data.edge_index)

    loss = F.cross_entropy(
        logits[data.val_mask],
        data.y[data.val_mask],
    ).item()

    predictions = logits.argmax(dim=1)

    accuracy = (
        predictions[data.val_mask]
        == data.y[data.val_mask]
    ).float().mean().item()

    return loss, accuracy


# ============================================================
# TRAINING
# ============================================================

def train_model(name, seed):

    set_seed(seed)

    model = create_model(name).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    best_loss = float("inf")
    best_accuracy = 0.0
    best_epoch = 0

    patience_counter = 0

    checkpoint_path = Path(
        f"models/{name.lower()}_best_seed{seed}.pt"
    )

    start_time = time.perf_counter()

    for epoch in range(1, MAX_EPOCHS + 1):

        model.train()
        optimizer.zero_grad()

        logits = model(data.x, data.edge_index)

        train_loss = F.cross_entropy(
            logits[data.train_mask],
            data.y[data.train_mask],
        )

        train_loss.backward()
        optimizer.step()

        val_loss, val_accuracy = validate(model)

        if val_loss < best_loss - 1e-6:

            best_loss = val_loss
            best_accuracy = val_accuracy
            best_epoch = epoch
            patience_counter = 0

            torch.save(
                {
                    "model_name": name,
                    "seed": seed,
                    "epoch": epoch,
                    "in_channels": dataset.num_features,
                    "hidden_channels": HIDDEN,
                    "out_channels": dataset.num_classes,
                    "dropout": DROPOUT,
                    "best_val_loss": best_loss,
                    "best_val_accuracy": best_accuracy,
                    "model_state_dict": model.state_dict(),
                },
                checkpoint_path,
            )

        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            break

    elapsed = time.perf_counter() - start_time

    return {
        "model": name,
        "seed": seed,
        "best_epoch": best_epoch,
        "val_loss": best_loss,
        "val_accuracy": best_accuracy,
        "training_seconds": round(elapsed, 3),
    }


# ============================================================
# MAIN EXPERIMENT
# ============================================================

results = []

print("=" * 65)
print("CORA - MULTI-SEED VALIDATION")
print("=" * 65)

print("New seeds:", SEEDS)
print("Existing seed: 42")
print("Models:", MODELS)
print("Device:", DEVICE)

for name in MODELS:

    print(f"\n===== {name} =====", flush=True)

    # Reuse the already-trained seed-42 checkpoint.
    existing_path = Path(
        f"models/{name.lower()}_best_seed42.pt"
    )

    checkpoint = torch.load(
        existing_path,
        map_location=DEVICE,
        weights_only=True,
    )

    results.append({
        "model": name,
        "seed": 42,
        "best_epoch": checkpoint["epoch"],
        "val_loss": checkpoint["best_val_loss"],
        "val_accuracy": checkpoint["best_val_accuracy"],
        "training_seconds": "",
    })

    print(
        f"Seed 42 | "
        f"Val Acc: {checkpoint['best_val_accuracy']:.4f}",
        flush=True,
    )

    for seed in SEEDS:

        result = train_model(name, seed)

        results.append(result)

        print(
            f"Seed {seed} | "
            f"Epoch: {result['best_epoch']} | "
            f"Val Acc: {result['val_accuracy']:.4f} | "
            f"Val Loss: {result['val_loss']:.4f} | "
            f"Time: {result['training_seconds']:.2f}s",
            flush=True,
        )


# ============================================================
# SAVE RESULTS
# ============================================================

with OUTPUT.open(
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
            "val_loss",
            "val_accuracy",
            "training_seconds",
        ],
    )

    writer.writeheader()
    writer.writerows(results)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("FINAL MULTI-SEED VALIDATION SUMMARY")
print("=" * 65)

for name in MODELS:

    values = np.array([
        row["val_accuracy"]
        for row in results
        if row["model"] == name
    ])

    print(
        f"{name:6s} | "
        f"Mean accuracy: {values.mean():.4f} | "
        f"Std: {values.std(ddof=1):.4f}"
    )

print("\nResults saved to:", OUTPUT)
print("Test set has not been evaluated.")
