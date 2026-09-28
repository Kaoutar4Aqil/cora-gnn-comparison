
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GATv2Conv


class GATv2Model(nn.Module):
    """
    Two-layer Graph Attention Network v2
    for semi-supervised node classification.
    """

    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        dropout=0.5,
        heads=8,
    ):
        super().__init__()

        if hidden_channels % heads != 0:
            raise ValueError(
                "hidden_channels must be divisible by heads."
            )

        self.dropout = dropout
        self.heads = heads

        channels_per_head = hidden_channels // heads

        # Layer 1: multi-head attention.
        # 8 heads x 8 channels = 64 hidden features.
        self.conv1 = GATv2Conv(
            in_channels=in_channels,
            out_channels=channels_per_head,
            heads=heads,
            concat=True,
            dropout=dropout,
        )

        # Layer 2: classification logits.
        # A single attention head outputs 7 classes.
        self.conv2 = GATv2Conv(
            in_channels=hidden_channels,
            out_channels=out_channels,
            heads=1,
            concat=False,
            dropout=dropout,
        )

    def forward(self, x, edge_index):

        x = F.dropout(
            x,
            p=self.dropout,
            training=self.training,
        )

        x = self.conv1(x, edge_index)

        x = F.elu(x)

        x = F.dropout(
            x,
            p=self.dropout,
            training=self.training,
        )

        x = self.conv2(x, edge_index)

        return x
