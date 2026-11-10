
"""Run one controlled SBM SNR diagnostic experiment."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from src.data.synthetic import generate_sbm_graph
from src.data.splits import stratified_split
from src.models.gcn import GCN, normalize_adj
from src.metrics import mean_pairwise_cosine_similarity


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_name):
    if device_name == "auto":
        return torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    return torch.device(device_name)


def accuracy(logits, labels):
    predictions = logits.argmax(dim=1)
    return (predictions == labels).float().mean().item()


def evaluate(model, x, adj_norm, labels, mask):
    model.eval()

    with torch.no_grad():
        logits = model(x, adj_norm)

        loss = F.cross_entropy(
            logits[mask],
            labels[mask],
        ).item()

        acc = accuracy(logits[mask], labels[mask])

    return loss, acc


def save_result(result, result_path):
    temp_path = result_path.with_suffix(".tmp")

    with open(temp_path, "w", encoding="utf-8") as file:
        json.dump(
            result,
            file,
            indent=4,
            ensure_ascii=False,
        )

    temp_path.replace(result_path)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run one SBM SNR diagnostic experiment."
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/diagnostics/snr.yaml",
    )

    parser.add_argument("--noise-scale", type=float, default=None)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )

    return parser


def main():
    args = build_parser().parse_args()

    # 1. Load configuration.
    with open(args.config, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    data_config = config["data"]
    model_config = config["model"]
    train_config = config["train"]

    noise_scale = (
        args.noise_scale
        if args.noise_scale is not None
        else data_config["noise_scale_list"][0]
    )

    num_layers = (
        args.num_layers
        if args.num_layers is not None
        else model_config["num_layers_list"][0]
    )

    seed = (
        args.seed
        if args.seed is not None
        else config["experiment"]["seed_list"][0]
    )

    if noise_scale not in data_config["noise_scale_list"]:
        raise ValueError("noise_scale is not in the configured list.")

    if num_layers not in model_config["num_layers_list"]:
        raise ValueError("num_layers is not in the configured list.")

    if seed not in config["experiment"]["seed_list"]:
        raise ValueError("seed is not in the configured list.")

    set_seed(seed)
    device = get_device(args.device)

    print(f"Device: {device}")
    print(f"Noise scale: {noise_scale}")
    print(f"GCN layers: {num_layers}")
    print(f"Seed: {seed}")

    # 2. Generate synthetic graph.
    adjacency, features, labels = generate_sbm_graph(
        num_communities=data_config["num_communities"],
        nodes_per_community=data_config["nodes_per_community"],
        feature_dim=data_config["feature_dim"],
        edge_prob_intra=data_config["edge_prob_intra"],
        edge_prob_inter=data_config["edge_prob_inter"],
        seed=seed,
        noise_scale=noise_scale,
    )

    # 3. Split nodes by community.
    train_mask, val_mask, test_mask = stratified_split(
        labels,
        train_ratio=train_config["train_ratio"],
        val_ratio=train_config["val_ratio"],
        test_ratio=train_config["test_ratio"],
        seed=seed,
    )

    # 4. Convert data to PyTorch tensors.
    x = torch.from_numpy(features).float().to(device)
    adj = torch.from_numpy(adjacency).float().to(device)
    y = torch.from_numpy(labels).long().to(device)

    train_mask = torch.from_numpy(train_mask).to(device)
    val_mask = torch.from_numpy(val_mask).to(device)
    test_mask = torch.from_numpy(test_mask).to(device)

    adj_norm = normalize_adj(adj)

    num_features = x.shape[1]
    num_classes = int(y.max().item()) + 1

    print(f"Nodes: {x.shape[0]}")
    print(f"Features: {num_features}")
    print(f"Classes: {num_classes}")
    print(f"Train nodes: {train_mask.sum().item()}")
    print(f"Validation nodes: {val_mask.sum().item()}")
    print(f"Test nodes: {test_mask.sum().item()}")

    # 5. Build GCN.
    model = GCN(
        in_dim=num_features,
        n_layers=num_layers,
        hidden=model_config["hidden_dim"],
        num_classes=num_classes,
        dropout=model_config["dropout"],
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_config["learning_rate"],
        weight_decay=train_config["weight_decay"],
    )

    # 6. Prepare output.
    output_dir = (
        Path(config["output"]["dir"])
        / f"noise_{noise_scale:g}"
        / f"layers_{num_layers}"
        / f"seed_{seed}"
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "result.json"

    result = {
        "experiment": config["experiment"]["name"],
        "noise_scale": noise_scale,
        "num_layers": num_layers,
        "seed": seed,
        "device": str(device),
        "epochs": train_config["epochs"],
        "history": [],
        "best_val_acc": None,
        "best_epoch": None,
        "test_acc": None,
        "status": "running",
    }

    # 7. Training loop.
    best_val_acc = -1.0
    best_epoch = 0
    best_state = None

    for epoch in range(1, train_config["epochs"] + 1):
        model.train()
        optimizer.zero_grad()

        logits = model(x, adj_norm)

        train_loss = F.cross_entropy(
            logits[train_mask],
            y[train_mask],
        )

        train_acc = accuracy(
            logits[train_mask],
            y[train_mask],
        )

        train_loss.backward()
        optimizer.step()

        val_loss, val_acc = evaluate(
            model,
            x,
            adj_norm,
            y,
            val_mask,
        )

        result["history"].append({
            "epoch": epoch,
            "train_loss": float(train_loss.item()),
            "train_acc": float(train_acc),
            "val_loss": float(val_loss),
            "val_acc": float(val_acc),
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch

            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

        if epoch % 20 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:03d}/{train_config['epochs']} "
                f"Train Acc: {train_acc:.4f} "
                f"Val Acc: {val_acc:.4f}"
            )

    # 8. Evaluate best validation checkpoint.
    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint.")

    model.load_state_dict(best_state)

    model.eval()

    with torch.no_grad():
        logits, hidden_states = model(
            x,
            adj_norm,
            return_hidden=True,
        )

        test_loss = F.cross_entropy(
            logits[test_mask],
            y[test_mask],
        ).item()

        test_acc = accuracy(
            logits[test_mask],
            y[test_mask],
        )

    oversmoothing_metrics = [
        {
            "layer": layer_index,
            "mean_pairwise_cosine_similarity": (
                mean_pairwise_cosine_similarity(hidden)
            ),
        }
        for layer_index, hidden in enumerate(hidden_states, start=1)
    ]

    result["oversmoothing_metrics"] = oversmoothing_metrics
    result["best_val_acc"] = float(best_val_acc)
    result["best_epoch"] = best_epoch
    result["test_loss"] = float(test_loss)
    result["test_acc"] = float(test_acc)
    result["status"] = "completed"

    # 9. Save result.
    save_result(result, result_path)

    print("\nTraining completed.")
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Result saved to: {result_path}")


if __name__ == "__main__":
    main()
