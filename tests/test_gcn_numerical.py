
import numpy as np
import torch

from src.models.gcn import GCNConv, normalize_adj


def test_gcn_conv_matches_numpy_reference():
    """Verify GCNConv against a fixed NumPy reference."""

    # 1. Fixed 5-node chain graph.
    A = np.array(
        [
            [0, 1, 0, 0, 0],
            [1, 0, 1, 0, 0],
            [0, 1, 0, 1, 0],
            [0, 0, 1, 0, 1],
            [0, 0, 0, 1, 0],
        ],
        dtype=np.float64,
    )

    # 2. Fixed node features.
    X = np.array(
        [
            [1, 0],
            [0, 1],
            [1, 1],
            [2, 0],
            [0, 2],
        ],
        dtype=np.float64,
    )

    # 3. Fixed weight and bias.
    W = np.array(
        [
            [1, 2],
            [-1, 1],
        ],
        dtype=np.float64,
    )

    b = np.array([0.5, -0.5], dtype=np.float64)

    # 4. NumPy reference calculation.
    I = np.eye(5, dtype=np.float64)
    A_hat = A + I

    degree = A_hat.sum(axis=1)
    D_inv_sqrt = np.diag(degree ** -0.5)

    A_norm = D_inv_sqrt @ A_hat @ D_inv_sqrt

    numpy_reference = A_norm @ X @ W + b

    # 5. Fixed expected result.
    expected = np.array(
        [
            [0.591751709536137, 0.908248290463863],
            [0.574914957130530, 1.649829914261059],
            [0.833333333333333, 2.166666666666667],
            [0.350170085738941, 2.649829914261059],
            [0.316496580927726, 2.132993161855453],
        ],
        dtype=np.float64,
    )

    # Check the NumPy calculation against the fixed reference.
    np.testing.assert_allclose(
        numpy_reference,
        expected,
        rtol=0.0,
        atol=1e-12,
    )

    # 6. Convert inputs to PyTorch tensors.
    A_tensor = torch.tensor(A, dtype=torch.float64)
    X_tensor = torch.tensor(X, dtype=torch.float64)

    # 7. Normalize adjacency using our own implementation.
    A_norm_tensor = normalize_adj(A_tensor)

    # 8. Create GCNConv.
    model = GCNConv(2, 2).double()

    # 9. Manually set weight and bias.
    with torch.no_grad():
        model.weight.copy_(torch.tensor(W))
        model.bias.copy_(torch.tensor(b))

    # 10. Forward propagation.
    output = model(X_tensor, A_norm_tensor)

    # 11. Calculate maximum absolute error.
    output_numpy = output.detach().cpu().numpy()

    max_error = np.max(
        np.abs(output_numpy - expected)
    )

    print(f"Maximum absolute error: {max_error:.12e}")

    # 12. Numerical alignment requirement.
    assert max_error < 1e-7
