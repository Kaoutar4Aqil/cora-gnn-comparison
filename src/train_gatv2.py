
# ============================================================
# CORA - GCN TRAINING
# Semi-supervised node classification
# ============================================================

import csv
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from torch_geometric.datasets import Planetoid

from gatv2_model import GATv2Model


# ============================================================
# 1. CONFIGURATION
# ============================================================

SEED = 42

HIDDEN_CHANNELS = 64
DROPOUT = 0.5

LEARNING_RATE = 0.01
WEIGHT_DECAY = 5e-4

MAX_EPOCHS = 300
PATIENCE = 50

DEVICE = torch.device("cpu")

CHECKPOINT_PATH = Path(
    "models/gatv2_best_seed42.pt"
)

METRICS_PATH = Path(
    "outputs/metrics/gatv2_training_seed42.csv"
)

CHECKPOINT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

METRICS_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 2. REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# 3. DATASET
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
# 4. MODEL
# ============================================================

model = GATv2Model(
    in_channels=dataset.num_features,
    hidden_channels=HIDDEN_CHANNELS,
    out_channels=dataset.num_classes,
    dropout=DROPOUT,
).to(DEVICE)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)

num_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)


# ============================================================
# 5. TRAINING FUNCTION
# ============================================================

def train_one_epoch():

    model.train()

    optimizer.zero_grad()

    logits = model(
        data.x,
        data.edge_index,
    )

    # Only training labels influence the gradient.
    loss = F.cross_entropy(
        logits[data.train_mask],
        data.y[data.train_mask],
    )

    loss.backward()

    optimizer.step()

    return float(loss.item())


# ============================================================
# 6. VALIDATION FUNCTION
# ============================================================

@torch.no_grad()
def evaluate_train_and_validation():

    model.eval()

    logits = model(
        data.x,
        data.edge_index,
    )

    predictions = logits.argmax(dim=1)

    train_loss = F.cross_entropy(
        logits[data.train_mask],
        data.y[data.train_mask],
    ).item()

    val_loss = F.cross_entropy(
        logits[data.val_mask],
        data.y[data.val_mask],
    ).item()

    train_accuracy = (
        predictions[data.train_mask]
        == data.y[data.train_mask]
    ).float().mean().item()

    val_accuracy = (
        predictions[data.val_mask]
        == data.y[data.val_mask]
    ).float().mean().item()

    return (
        train_loss,
        val_loss,
        train_accuracy,
        val_accuracy,
    )


# ============================================================
# 7. TRAINING LOOP
# ============================================================

print("=" * 65)
print("CORA - GATv2 TRAINING")
print("=" * 65)

print(f"\nDevice: {DEVICE}")
print(f"Random seed: {SEED}")
print(f"Training nodes: {int(data.train_mask.sum())}")
print(f"Validation nodes: {int(data.val_mask.sum())}")
print(f"Model parameters: {num_parameters}")

print(f"\nHidden channels: {HIDDEN_CHANNELS}")
print(f"Dropout: {DROPOUT}")
print(f"Learning rate: {LEARNING_RATE}")
print(f"Weight decay: {WEIGHT_DECAY}")

print(f"Maximum epochs: {MAX_EPOCHS}")
print(f"Early stopping patience: {PATIENCE}")

best_val_loss = float("inf")
best_val_accuracy = 0.0
best_epoch = 0

epochs_without_improvement = 0
history = []

start_time = time.perf_counter()

for epoch in range(1, MAX_EPOCHS + 1):

    train_one_epoch()

    (
        train_loss,
        val_loss,
        train_accuracy,
        val_accuracy,
    ) = evaluate_train_and_validation()

    history.append({
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "train_accuracy": train_accuracy,
        "val_accuracy": val_accuracy,
    })

    # Select the checkpoint using validation loss.
    # Never use test labels during model selection.
    if val_loss < best_val_loss - 1e-6:

        best_val_loss = val_loss
        best_val_accuracy = val_accuracy
        best_epoch = epoch

        epochs_without_improvement = 0

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "model_name": "GATv2",
                "seed": SEED,
                "epoch": epoch,
                "in_channels": dataset.num_features,
                "hidden_channels": HIDDEN_CHANNELS,
                "out_channels": dataset.num_classes,
                "dropout": DROPOUT,
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "best_val_loss": best_val_loss,
                "best_val_accuracy": best_val_accuracy,
            },
            CHECKPOINT_PATH,
        )

    else:

        epochs_without_improvement += 1

    if epoch == 1 or epoch % 20 == 0:

        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Train Acc: {train_accuracy:.4f} | "
            f"Val Acc: {val_accuracy:.4f}"
        )

    if epochs_without_improvement >= PATIENCE:

        print(
            f"\nEarly stopping triggered at epoch {epoch}."
        )
        break


elapsed = time.perf_counter() - start_time


# ============================================================
# 8. SAVE TRAINING HISTORY
# ============================================================

with METRICS_PATH.open(
    "w",
    newline="",
    encoding="utf-8",
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "epoch",
            "train_loss",
            "val_loss",
            "train_accuracy",
            "val_accuracy",
        ],
    )

    writer.writeheader()
    writer.writerows(history)


# ============================================================
# 9. SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("GATv2 TRAINING COMPLETED")
print("=" * 65)

print(f"Epochs completed: {len(history)}")
print(f"Best epoch: {best_epoch}")
print(f"Best validation loss: {best_val_loss:.4f}")
print(f"Validation accuracy at best epoch: {best_val_accuracy:.4f}")
print(f"Training time: {elapsed:.2f} seconds")

print(f"\nBest checkpoint: {CHECKPOINT_PATH}")
print(f"Training metrics: {METRICS_PATH}")

print("\nTest set has not been evaluated.")
