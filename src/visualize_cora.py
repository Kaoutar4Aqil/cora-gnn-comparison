
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch

from matplotlib.patches import Patch
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import degree, k_hop_subgraph


# ============================================================
# CONFIGURATION
# ============================================================

OUTPUT_DIR = Path("outputs/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

dataset = Planetoid(
    root="data/Planetoid",
    name="Cora",
    split="public",
)

data = dataset[0]

# ============================================================
# FIGURE 1: CLASS DISTRIBUTION
# ============================================================

classes = np.arange(dataset.num_classes)

train_counts = []
validation_counts = []

for class_id in classes:

    class_nodes = data.y == int(class_id)

    train_counts.append(
        int((class_nodes & data.train_mask).sum())
    )

    validation_counts.append(
        int((class_nodes & data.val_mask).sum())
    )

fig, ax = plt.subplots(figsize=(10, 5.5))

width = 0.36

ax.bar(
    classes - width / 2,
    train_counts,
    width,
    label="Training",
    color="#168A84",
)

ax.bar(
    classes + width / 2,
    validation_counts,
    width,
    label="Validation",
    color="#4589D1",
)

ax.set_title(
    "Cora: Class Distribution",
    fontsize=15,
    fontweight="bold",
)

ax.set_xlabel("Class ID")
ax.set_ylabel("Number of labeled nodes")
ax.set_xticks(classes)
ax.legend()
ax.grid(axis="y", alpha=0.2)
ax.set_axisbelow(True)

fig.tight_layout()

fig.savefig(
    OUTPUT_DIR / "cora_class_distribution.png",
    dpi=180,
)

plt.close(fig)

# ============================================================
# FIGURE 2: LOCAL CITATION GRAPH
# ============================================================

node_degrees = degree(
    data.edge_index[0],
    num_nodes=data.num_nodes,
)

# Choose a node with a moderate degree for clarity.
center = int(
    torch.argmin(torch.abs(node_degrees - 6))
)

subset, local_edges, mapping, _ = k_hop_subgraph(
    center,
    num_hops=2,
    edge_index=data.edge_index,
    relabel_nodes=True,
    num_nodes=data.num_nodes,
)

graph = nx.Graph()

graph.add_nodes_from(range(len(subset)))

graph.add_edges_from(
    local_edges.t().tolist()
)

node_colors = []

palette = {
    "train": "#168A84",
    "validation": "#4589D1",
    "test": "#AD82DC",
    "other": "#AAB4BF",
}

for original_id in subset.tolist():

    if data.train_mask[original_id]:
        node_colors.append(palette["train"])

    elif data.val_mask[original_id]:
        node_colors.append(palette["validation"])

    elif data.test_mask[original_id]:
        node_colors.append(palette["test"])

    else:
        node_colors.append(palette["other"])

center_local = int(mapping.item())

node_sizes = [
    300 if i == center_local else 85
    for i in range(len(subset))
]

positions = nx.spring_layout(
    graph,
    seed=42,
    iterations=60,
)

fig, ax = plt.subplots(figsize=(11, 8))

nx.draw_networkx_edges(
    graph,
    positions,
    ax=ax,
    alpha=0.22,
    width=0.8,
)

nx.draw_networkx_nodes(
    graph,
    positions,
    ax=ax,
    node_color=node_colors,
    node_size=node_sizes,
    edgecolors="white",
    linewidths=0.5,
)

nx.draw_networkx_labels(
    graph,
    positions,
    labels={center_local: str(center)},
    font_size=9,
    ax=ax,
)

legend_items = [
    Patch(color=color, label=group.title())
    for group, color in palette.items()
]

ax.legend(
    handles=legend_items,
    loc="upper right",
)

ax.set_title(
    "Cora Citation Graph: 2-Hop Neighborhood",
    fontsize=15,
    fontweight="bold",
)

ax.axis("off")
fig.tight_layout()

fig.savefig(
    OUTPUT_DIR / "cora_graph_neighborhood.png",
    dpi=180,
)

plt.close(fig)

print("=" * 60)
print("CORA VISUALIZATION COMPLETED")
print("=" * 60)

print("Center node:", center)
print("Nodes in local subgraph:", len(subset))
print("Edges in local subgraph:", graph.number_of_edges())

print("\nGenerated figures:")
print("outputs/figures/cora_class_distribution.png")
print("outputs/figures/cora_graph_neighborhood.png")
