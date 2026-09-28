import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import APPNP


class APPNPModel(nn.Module):
    """
    MLP followed by APPNP propagation
    for semi-supervised node classification.
    """

    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        dropout=0.5,
        propagation_steps=10,
        alpha=0.1,
    ):
        super().__init__()

        self.dropout = dropout

        self.lin1 = nn.Linear(
            in_channels,
            hidden_channels,
        )

        self.lin2 = nn.Linear(
            hidden_channels,
            out_channels,
        )

        self.propagation = APPNP(
            K=propagation_steps,
            alpha=alpha,
            dropout=0.0,
        )

    def forward(self, x, edge_index):

        x = F.dropout(
            x,
            p=self.dropout,
            training=self.training,
        )

        x = self.lin1(x)
        x = F.relu(x)

        x = F.dropout(
            x,
            p=self.dropout,
            training=self.training,
        )

        x = self.lin2(x)

        x = self.propagation(
            x,
            edge_index,
        )

        return x
