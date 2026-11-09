
"""Boundary and property tests for GCN."""

import random

import numpy as np
import torch
import torch.nn.functional as F

from src.models.gcn import GCN, GCNConv, normalize_adj


def test_normalize_adj_is_symmetric_but_not_row_normalized():
    """Symmetric normalization should not be confused with row normalization."""

    adj = torch.tensor(
        [
            [0, 1, 1, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 1],
            [0, 0, 1, 0],
        ],
        dtype=torch.float64,
    )

    adj_norm = normalize_adj(adj)

    # 1. Symmetric normalization should produce a symmetric matrix.
    assert torch.allclose(
        adj_norm,
        adj_norm.T,
        atol=1e-12,
    )

    # 2. Symmetric normalization does not generally make every row sum to 1.
    row_sums = adj_norm.sum(dim=1)

    assert not torch.allclose(
        row_sums,
        torch.ones_like(row_sums),
        atol=1e-6,
    )


def test_gcnconv_preserves_node_dimension_and_output_shape():
    """GCN propagation should preserve the number of nodes."""

    num_nodes = 8
    in_dim = 3
    out_dim = 5

    x = torch.randn(num_nodes, in_dim)

    adj = torch.eye(num_nodes)
    adj_norm = normalize_adj(adj)

    conv = GCNConv(in_dim, out_dim)
    output = conv(x, adj_norm)

    assert output.shape == (num_nodes, out_dim)


def test_isolated_node_does_not_produce_nan():
    """An isolated node should have finite normalized values and outputs."""

    adj = torch.tensor(
        [
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, 0],
        ],
        dtype=torch.float32,
    )

    adj_norm = normalize_adj(adj)

    # The isolated node should not produce NaN or Inf.
    assert torch.isfinite(adj_norm).all()

    # After adding self-loops, the isolated node has a self-connection.
    assert torch.isclose(
        adj_norm[2, 2],
        torch.tensor(1.0),
    )

    x = torch.randn(3, 4)
    conv = GCNConv(4, 2)

    output = conv(x, adj_norm)

    assert torch.isfinite(output).all()


def test_one_layer_gcn_is_equivalent_to_linear_classifier():
    """A one-layer GCN with identity adjacency should be an affine classifier."""

    x = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [2.0, 1.0, 0.0],
            [0.0, 1.0, 2.0],
        ]
    )

    num_nodes = x.shape[0]

    # Identity adjacency means no information is exchanged between nodes.
    adj = torch.eye(num_nodes)
    adj_norm = normalize_adj(adj)

    model = GCN(
        in_dim=3,
        n_layers=1,
        hidden=8,
        num_classes=2,
    )

    output = model(x, adj_norm)

    # A one-layer GCN should contain exactly one convolution.
    assert len(model.convs) == 1

    # With identity normalized adjacency:
    # output = I @ X @ W + b = X @ W + b
    conv = model.convs[0]

    expected = x @ conv.weight + conv.bias

    assert torch.allclose(
        output,
        expected,
        atol=1e-6,
    )


def _train_once(seed):
    """Run a small deterministic training experiment."""

    # Fix random seeds.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Use a fixed toy graph.
    adj = torch.tensor(
        [
            [0, 1, 0, 0, 0, 0],
            [1, 0, 1, 0, 0, 0],
            [0, 1, 0, 1, 0, 0],
            [0, 0, 1, 0, 1, 0],
            [0, 0, 0, 1, 0, 1],
            [0, 0, 0, 0, 1, 0],
        ],
        dtype=torch.float32,
    )

    # Random features are reproducible because the seed is fixed.
    x = torch.randn(6, 4)

    y = torch.tensor(
        [0, 0, 0, 1, 1, 1],
        dtype=torch.long,
    )

    adj_norm = normalize_adj(adj)

    model = GCN(
        in_dim=4,
        n_layers=2,
        hidden=8,
        num_classes=2,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.01,
        weight_decay=0.0005,
    )

    losses = []

    # A small full-batch training loop.
    for _ in range(10):
        model.train()
        optimizer.zero_grad()

        logits = model(x, adj_norm)
        loss = F.cross_entropy(logits, y)

        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    # Save the final model parameters.
    final_state = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
    }

    return losses, final_state


def test_same_seed_produces_identical_training_results():
    """Two training runs with the same seed should produce identical results."""

    losses_1, state_1 = _train_once(seed=42)
    losses_2, state_2 = _train_once(seed=42)

    # Training loss history should be identical.
    assert np.allclose(
        losses_1,
        losses_2,
        rtol=0.0,
        atol=0.0,
    )

    # Final model parameters should also be identical.
    assert state_1.keys() == state_2.keys()

    for name in state_1:
        assert torch.equal(
            state_1[name],
            state_2[name],
        ), f"Parameters differ: {name}"
