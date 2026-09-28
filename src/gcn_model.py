
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv


class GCNModel(nn.Module):
    """
    Two-layer Graph Convolutional Network
    for semi-supervised node classification.
    """

    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        dropout=0.5,
    ):
        super().__init__()

        self.dropout = dropout

        self.conv1 = GCNConv(
            in_channels,
            hidden_channels,
        )

        self.conv2 = GCNConv(
            hidden_channels,
            out_channels,
        )

    def forward(self, x, edge_index):

        # Input feature dropout
        x = F.dropout(
            x,
            p=self.dropout,
            training=self.training,
        )

        # First graph convolution
        x = self.conv1(x, edge_index)

        x = F.relu(x)

        # Hidden representation dropout
        x = F.dropout(
            x,
            p=self.dropout,
            training=self.training,
        )

        # Second graph convolution
        x = self.conv2(x, edge_index)

        # Raw logits for 7 classes
        return x
