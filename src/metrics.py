import torch
import torch.nn.functional as F


def mean_pairwise_cosine_similarity(
    embeddings: torch.Tensor,
) -> float:
    """Compute mean cosine similarity between distinct node embeddings."""

    if embeddings.ndim != 2:
        raise ValueError("embeddings must be a 2-D tensor.")

    num_nodes = embeddings.shape[0]

    if num_nodes < 2:
        raise ValueError("At least two nodes are required.")

    if not torch.is_floating_point(embeddings):
        raise ValueError("embeddings must be floating-point.")

    if not torch.isfinite(embeddings).all():
        raise ValueError("embeddings must contain only finite values.")

    normalized = F.normalize(
        embeddings,
        p=2,
        dim=1,
        eps=1e-12,
    )

    similarity = normalized @ normalized.T

    mask = ~torch.eye(
        num_nodes,
        dtype=torch.bool,
        device=embeddings.device,
    )

    return similarity[mask].mean().item()