
"""Tests for the synthetic SBM graph generator."""

import numpy as np

from src.data.synthetic import generate_sbm_graph


def count_connected_components(adjacency: np.ndarray) -> int:
    """Count connected components using DFS."""
    num_nodes = adjacency.shape[0]
    visited = np.zeros(num_nodes, dtype=bool)
    component_count = 0

    for start in range(num_nodes):
        if visited[start]:
            continue

        component_count += 1
        stack = [start]
        visited[start] = True

        while stack:
            node = stack.pop()

            neighbors = np.flatnonzero(adjacency[node])

            for neighbor in neighbors:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append(int(neighbor))

    return component_count


def test_synthetic_graph_shapes():
    """Test output shapes and isolated node behavior."""
    k = 4
    nodes_per_community = 20
    feature_dim = 16

    adjacency, features, labels = generate_sbm_graph(
        num_communities=k,
        nodes_per_community=nodes_per_community,
        feature_dim=feature_dim,
        edge_prob_intra=0.2,
        edge_prob_inter=0.01,
        seed=42,
        isolated_node_indices=[0],
    )

    num_nodes = k * nodes_per_community

    assert adjacency.shape == (num_nodes, num_nodes)
    assert features.shape == (num_nodes, feature_dim)
    assert labels.shape == (num_nodes,)

    assert adjacency.dtype == np.uint8
    assert features.dtype == np.float32
    assert labels.dtype == np.int64

    # Adjacency matrix must be symmetric.
    assert np.array_equal(adjacency, adjacency.T)

    # No self-loops.
    assert np.all(np.diag(adjacency) == 0)

    # Node 0 must be isolated.
    assert np.sum(adjacency[0]) == 0
    assert np.sum(adjacency[:, 0]) == 0

    # Every community has the expected number of nodes.
    assert np.array_equal(
        np.bincount(labels),
        np.full(k, nodes_per_community),
    )


def test_connected_components_equal_communities():
    """Test that k disconnected communities produce k components."""
    k = 4
    nodes_per_community = 30

    adjacency, _, _ = generate_sbm_graph(
        num_communities=k,
        nodes_per_community=nodes_per_community,
        feature_dim=16,
        edge_prob_intra=1.0,
        edge_prob_inter=0.0,
        seed=42,
    )

    component_count = count_connected_components(adjacency)

    assert component_count == k


def test_same_seed_produces_identical_graph():
    """Test byte-level reproducibility for the same seed."""
    kwargs = dict(
        num_communities=4,
        nodes_per_community=20,
        feature_dim=16,
        edge_prob_intra=0.2,
        edge_prob_inter=0.01,
        seed=123,
        isolated_node_indices=[0, 5],
    )

    result1 = generate_sbm_graph(**kwargs)
    result2 = generate_sbm_graph(**kwargs)

    for array1, array2 in zip(result1, result2):
        assert array1.tobytes() == array2.tobytes()
