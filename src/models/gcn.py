
import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize_adj(adj: torch.Tensor) -> torch.Tensor:
    """Add self-loops and symmetrically normalize adjacency."""

    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError("adj must be a square matrix.")

    if not torch.is_floating_point(adj):
        adj = adj.to(torch.float32)

    num_nodes = adj.shape[0]

    identity = torch.eye(
        num_nodes,
        dtype=adj.dtype,
        device=adj.device,
    )

    adj_hat = adj + identity
    degree = adj_hat.sum(dim=1)
    degree_inv_sqrt = degree.clamp_min(1e-12).pow(-0.5)

    adj_norm = (
        degree_inv_sqrt.unsqueeze(1)
        * adj_hat
        * degree_inv_sqrt.unsqueeze(0)
    )

    return adj_norm


class GCNConv(nn.Module):
    """A hand-written GCN convolution layer."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()

        if in_dim < 1 or out_dim < 1:
            raise ValueError("Feature dimensions must be positive.")

        self.weight = nn.Parameter(
            torch.empty(in_dim, out_dim)
        )

        self.bias = nn.Parameter(
            torch.zeros(out_dim)
        )

        nn.init.xavier_uniform_(self.weight)

    def forward(
        self,
        x: torch.Tensor,
        adj_norm: torch.Tensor,
    ) -> torch.Tensor:

        if x.ndim != 2:
            raise ValueError("x must be a 2-D tensor.")

        if adj_norm.ndim != 2:
            raise ValueError("adj_norm must be a 2-D tensor.")

        if adj_norm.shape[0] != adj_norm.shape[1]:
            raise ValueError("adj_norm must be square.")

        if adj_norm.shape[0] != x.shape[0]:
            raise ValueError("Node dimensions do not match.")

        if x.shape[1] != self.weight.shape[0]:
            raise ValueError("Input feature dimension mismatch.")

        support = x @ self.weight
        output = adj_norm @ support + self.bias

        return output


class GCN(nn.Module):
    """Multi-layer Graph Convolutional Network."""

    def __init__(
        self,
        in_dim: int,
        n_layers: int,
        hidden: int,
        num_classes: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        if in_dim < 1 or hidden < 1 or num_classes < 1:
            raise ValueError("All dimensions must be positive.")

        if n_layers < 1:
            raise ValueError("n_layers must be >= 1.")

        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")

        self.n_layers = n_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()

        if n_layers == 1:
            self.convs.append(
                GCNConv(in_dim, num_classes)
            )
        else:
            self.convs.append(
                GCNConv(in_dim, hidden)
            )

            for _ in range(n_layers - 2):
                self.convs.append(
                    GCNConv(hidden, hidden)
                )

            self.convs.append(
                GCNConv(hidden, num_classes)
            )

    def forward(
        self,
        x: torch.Tensor,
        adj_norm: torch.Tensor,
        return_hidden: bool = False,
    ):
        hidden_states = []

        for i, conv in enumerate(self.convs):

            x = F.dropout(
                x,
                p=self.dropout,
                training=self.training,
            )

            x = conv(x, adj_norm)

            if i < self.n_layers - 1:
                x = F.relu(x)
                hidden_states.append(x)

        if return_hidden:
            return x, hidden_states

        return x
