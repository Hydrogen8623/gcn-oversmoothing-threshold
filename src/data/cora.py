
"""Cora dataset loading utilities."""

from pathlib import Path

import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_dense_adj, to_undirected


def load_cora(root: str = "data/Planetoid") -> dict:
    """Load Cora and convert its graph to a dense adjacency matrix.

    Returns:
        Dictionary containing:
            x: Node features, shape (N, F).
            adj: Original adjacency matrix, shape (N, N).
            y: Node labels, shape (N,).
            train_mask: Training node mask.
            val_mask: Validation node mask.
            test_mask: Test node mask.
    """

    dataset = Planetoid(
        root=str(Path(root)),
        name="Cora",
    )

    data = dataset[0]

    # Ensure the graph is undirected.
    edge_index = to_undirected(
        data.edge_index,
        num_nodes=data.num_nodes,
    )

    # Convert edge_index to a dense adjacency matrix.
    adj = to_dense_adj(
        edge_index,
        max_num_nodes=data.num_nodes,
    )[0]

    return {
        "x": data.x.to(torch.float32),
        "adj": adj.to(torch.float32),
        "y": data.y.to(torch.long),
        "train_mask": data.train_mask,
        "val_mask": data.val_mask,
        "test_mask": data.test_mask,
    }
