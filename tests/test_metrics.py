import pytest
import torch

from src.metrics import mean_pairwise_cosine_similarity


def test_identical_embeddings_have_similarity_one():
    embeddings = torch.tensor([
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
    ])

    result = mean_pairwise_cosine_similarity(embeddings)

    assert result == pytest.approx(1.0)


def test_orthogonal_embeddings_have_similarity_zero():
    embeddings = torch.tensor([
        [1.0, 0.0],
        [0.0, 1.0],
        [-1.0, 0.0],
    ])

    result = mean_pairwise_cosine_similarity(embeddings)

    assert result == pytest.approx(-1.0 / 3.0)


def test_zero_embeddings_do_not_produce_nan():
    embeddings = torch.tensor([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
    ])

    result = mean_pairwise_cosine_similarity(embeddings)

    assert result == pytest.approx(0.0)


def test_requires_at_least_two_nodes():
    embeddings = torch.tensor([[1.0, 0.0]])

    with pytest.raises(ValueError):
        mean_pairwise_cosine_similarity(embeddings)


def test_rejects_non_finite_embeddings():
    embeddings = torch.tensor([
        [1.0, 0.0],
        [float("nan"), 1.0],
    ])

    with pytest.raises(ValueError):
        mean_pairwise_cosine_similarity(embeddings)