
"""Synthetic graph generation using the Stochastic Block Model (SBM)."""

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt


def generate_sbm_graph(
    num_communities: int,
    nodes_per_community: int,
    feature_dim: int,
    edge_prob_intra: float,
    edge_prob_inter: float,
    seed: int,
    isolated_node_indices: Sequence[int] | None = None,
    noise_scale: float = 0.1,
) -> tuple[
    npt.NDArray[np.uint8],
    npt.NDArray[np.float32],
    npt.NDArray[np.int64],
]:
    """Generate an undirected SBM graph.

    Each community has the same number of nodes.

    Args:
        num_communities: Number of communities (k).
        nodes_per_community: Number of nodes in each community.
        feature_dim: Dimension of node features.
        edge_prob_intra: Edge probability within a community.
        edge_prob_inter: Edge probability between communities.
        seed: Random seed for reproducibility.
        isolated_node_indices: Node indices that should have no edges.
        noise_scale: Standard deviation of Gaussian feature noise.

    Returns:
        adjacency: Symmetric binary adjacency matrix, shape (N, N).
        features: Node feature matrix, shape (N, feature_dim).
        labels: Community labels, shape (N,).
    """
    # 1. Validate parameters.
    if num_communities < 1:
        raise ValueError("num_communities must be >= 1.")

    if nodes_per_community < 1:
        raise ValueError("nodes_per_community must be >= 1.")

    if feature_dim < 1:
        raise ValueError("feature_dim must be >= 1.")

    if not np.isfinite(noise_scale) or noise_scale < 0:
        raise ValueError("noise_scale must be finite and >= 0.")

    for name, probability in (
        ("edge_prob_intra", edge_prob_intra),
        ("edge_prob_inter", edge_prob_inter),
    ):
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1.")

    # Use a local RNG. Never modify NumPy's global random state.
    rng = np.random.default_rng(seed)

    # 2. Generate community labels.
    num_nodes = num_communities * nodes_per_community

    labels = np.repeat(
        np.arange(num_communities, dtype=np.int64),
        nodes_per_community,
    )

    # 3. Generate adjacency matrix.
    same_community = labels[:, None] == labels[None, :]

    probabilities = np.where(
        same_community,
        edge_prob_intra,
        edge_prob_inter,
    )

    # Generate each undirected edge only once.
    random_values = rng.random((num_nodes, num_nodes))

    upper_triangle = np.triu(
        random_values < probabilities,
        k=1,
    )

    adjacency = (
        upper_triangle | upper_triangle.T
    ).astype(np.uint8)

    # 4. Explicitly create isolated nodes.
    if isolated_node_indices is not None:
        isolated_indices = np.asarray(
            list(isolated_node_indices),
            dtype=np.int64,
        )

        if isolated_indices.ndim != 1:
            raise ValueError("isolated_node_indices must be 1-D.")

        if np.any(isolated_indices < 0) or np.any(
            isolated_indices >= num_nodes
        ):
            raise ValueError("Isolated node index out of range.")

        if len(np.unique(isolated_indices)) != len(isolated_indices):
            raise ValueError("Duplicate isolated node indices.")

        adjacency[isolated_indices, :] = 0
        adjacency[:, isolated_indices] = 0

    # 5. Generate community-informed node features.
    #
    # Each community has its own feature prototype.
    # Nodes in the same community receive similar features.
    prototypes = rng.normal(
        loc=0.0,
        scale=1.0,
        size=(num_communities, feature_dim),
    )

    # Normalize prototypes.
    norms = np.linalg.norm(prototypes, axis=1, keepdims=True)
    prototypes = prototypes / np.maximum(norms, 1e-12)

    noise = rng.normal(
        loc=0.0,
        scale=noise_scale,
        size=(num_nodes, feature_dim),
    )

    features = (
        prototypes[labels] + noise
    ).astype(np.float32)

    return adjacency, features, labels
