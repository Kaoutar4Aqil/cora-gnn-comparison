
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from pathlib import Path
from sklearn.metrics import confusion_matrix
from torch_geometric.datasets import Planetoid

from gcn_model import GCNModel
from gatv2_model import GATv2Model
from appnp_model import APPNPModel


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42
DEVICE = torch.device("cpu")

OUTPUT = Path(
    "outputs/figures/confusion_matrices_seed42.png"
)

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# LOAD CORA
# ============================================================

dataset = Planetoid(
    root="data/Planetoid",
    name="Cora",
    split="public",
)

data = dataset[0].to(DEVICE)

assert int(data.test_mask.sum()) == 1000

class_ids = list(range(dataset.num_classes))

y_true = (
    data.y[data.test_mask]
    .cpu()
    .numpy()
)


# ============================================================
# MODEL FACTORY
# ============================================================

def build_model(name, checkpoint):

    kwargs = {
        "in_channels": checkpoint["in_channels"],
        "hidden_channels": checkpoint["hidden_channels"],
        "out_channels": checkpoint["out_channels"],
        "dropout": checkpoint["dropout"],
    }

    if name == "GCN":
        return GCNModel(**kwargs)

    if name == "GATv2":
        return GATv2Model(
            **kwargs,
            heads=8,
        )

    if name == "APPNP":
        return APPNPModel(
            **kwargs,
            propagation_steps=10,
            alpha=0.1,
        )

    raise ValueError(
        f"Unknown model: {name}"
    )


# ============================================================
# GENERATE CONFUSION MATRICES
# ============================================================

model_names = ["GCN", "GATv2", "APPNP"]

fig, axes = plt.subplots(
    1,
    3,
    figsize=(18, 5.8),
    constrained_layout=True,
)

for ax, name in zip(axes, model_names):

    checkpoint_path = Path(
        f"models/{name.lower()}_best_seed{SEED}.pt"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=True,
    )

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

    y_pred = (
        predictions[data.test_mask]
        .cpu()
        .numpy()
    )

    # Raw confusion matrix
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=class_ids,
    )

    # Normalize each row by its class support.
    cm_normalized = confusion_matrix(
        y_true,
        y_pred,
        labels=class_ids,
        normalize="true",
    )

    image = ax.imshow(
        cm_normalized,
        cmap="Blues",
        vmin=0,
        vmax=1,
    )

    ax.set_title(
        f"{name} - Seed {SEED}",
        fontsize=14,
        fontweight="bold",
    )

    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")

    ax.set_xticks(class_ids)
    ax.set_yticks(class_ids)

    # Display normalized values in each cell.
    for i in range(len(class_ids)):
        for j in range(len(class_ids)):

            value = cm_normalized[i, j]

            ax.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=9,
                color="white" if value > 0.5 else "black",
            )

    correct = int(np.trace(cm))
    total = int(cm.sum())

    print(
        f"{name}: {correct}/{total} "
        f"correctly classified "
        f"({correct / total:.4f})"
    )


# ============================================================
# SAVE FIGURE
# ============================================================

fig.colorbar(
    image,
    ax=axes,
    shrink=0.75,
    label="Proportion within true class",
)

fig.suptitle(
    "Cora - Normalized Test Confusion Matrices",
    fontsize=17,
    fontweight="bold",
)

fig.savefig(
    OUTPUT,
    dpi=200,
    bbox_inches="tight",
)

plt.close(fig)

print("\n" + "=" * 60)
print("CONFUSION MATRIX VISUALIZATION COMPLETED")
print("=" * 60)

print(f"Output: {OUTPUT}")
print("Test results used for reporting only.")
