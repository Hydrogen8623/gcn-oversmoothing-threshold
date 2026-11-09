
import torch

from src.models.gcn import normalize_adj, GCNConv, GCN


def test_normalize_adj_with_isolated_node():
    adj = torch.tensor([
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, 0],
    ], dtype=torch.float32)

    adj_norm = normalize_adj(adj)

    assert adj_norm.shape == (3, 3)
    assert torch.isfinite(adj_norm).all()

    # Isolated node becomes a self-loop-only node.
    assert torch.isclose(
        adj_norm[2, 2],
        torch.tensor(1.0),
    )


def test_gcnconv_output_shape():
    x = torch.randn(10, 16)
    adj = torch.eye(10)
    adj_norm = normalize_adj(adj)

    conv = GCNConv(16, 32)
    output = conv(x, adj_norm)

    assert output.shape == (10, 32)


def test_gcn_output_shape():
    x = torch.randn(100, 16)
    adj = torch.eye(100)
    adj_norm = normalize_adj(adj)

    model = GCN(
        in_dim=16,
        n_layers=4,
        hidden=64,
        num_classes=4,
    )

    output = model(x, adj_norm)

    assert output.shape == (100, 4)
    assert torch.isfinite(output).all()


def test_one_layer_gcn():
    x = torch.randn(10, 16)
    adj_norm = normalize_adj(torch.eye(10))

    model = GCN(
        in_dim=16,
        n_layers=1,
        hidden=64,
        num_classes=4,
    )

    output = model(x, adj_norm)

    assert output.shape == (10, 4)
