
import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import is_undirected

print("=" * 65)
print("CORA DATASET - EXPLORATORY ANALYSIS")
print("=" * 65)

# --------------------------------------------------
# 1. Load Cora using the standard public split
# --------------------------------------------------

dataset = Planetoid(
    root="data/Planetoid",
    name="Cora",
    split="public",
)

graph = dataset[0]

# --------------------------------------------------
# 2. General information
# --------------------------------------------------

print("\nGENERAL INFORMATION")

print("Number of graphs:", len(dataset))
print("Number of nodes:", graph.num_nodes)
print("Number of edges:", graph.num_edges)
print("Number of features:", dataset.num_features)
print("Number of classes:", dataset.num_classes)

print("\nTENSOR DIMENSIONS")

print("Node features (x):", tuple(graph.x.shape))
print("Edge index:", tuple(graph.edge_index.shape))
print("Node labels (y):", tuple(graph.y.shape))

# --------------------------------------------------
# 3. Train / validation / test split
# --------------------------------------------------

print("\nPUBLIC DATA SPLIT")

train_count = int(graph.train_mask.sum())
val_count = int(graph.val_mask.sum())
test_count = int(graph.test_mask.sum())

print("Training nodes:", train_count)
print("Validation nodes:", val_count)
print("Test nodes:", test_count)

assert not torch.any(
    graph.train_mask & graph.val_mask
)

assert not torch.any(
    graph.train_mask & graph.test_mask
)

assert not torch.any(
    graph.val_mask & graph.test_mask
)

print("Masks are disjoint: YES")

# --------------------------------------------------
# 4. Graph structure
# --------------------------------------------------

print("\nGRAPH STRUCTURE")

print(
    "Is undirected:",
    is_undirected(graph.edge_index),
)

print(
    "Average recorded edges per node:",
    round(graph.num_edges / graph.num_nodes, 2),
)

# --------------------------------------------------
# 5. Class distribution
# --------------------------------------------------

print("\nCLASS DISTRIBUTION")

for class_id in range(dataset.num_classes):

    total = int((graph.y == class_id).sum())

    train = int(
        ((graph.y == class_id) & graph.train_mask).sum()
    )

    val = int(
        ((graph.y == class_id) & graph.val_mask).sum()
    )

    print(
        f"Class {class_id}: "
        f"total={total}, "
        f"train={train}, "
        f"validation={val}"
    )

# --------------------------------------------------
# 6. Example node
# --------------------------------------------------

print("\nEXAMPLE NODE")

print("Node ID: 0")
print("Class ID:", int(graph.y[0]))
print(
    "Number of nonzero features:",
    int(torch.count_nonzero(graph.x[0])),
)

print("\nCora exploration completed successfully!")
