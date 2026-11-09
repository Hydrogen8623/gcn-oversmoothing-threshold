
"""Train a two-layer GCN on Cora."""

import argparse
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from src.data.cora import load_cora
from src.models.gcn import GCN, normalize_adj


def build_parser():
    parser = argparse.ArgumentParser(
        description="Train a GCN on Cora."
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/cora.yaml",
        help="Path to YAML config.",
    )

    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional experiment run ID.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Training device.",
    )

    return parser


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
    return (
        predictions == labels
    ).float().mean().item()


def evaluate(model, x, adj_norm, labels, mask):
    model.eval()

    with torch.no_grad():
        logits = model(x, adj_norm)

        loss = F.cross_entropy(
            logits[mask],
            labels[mask],
        ).item()

        acc = accuracy(
            logits[mask],
            labels[mask],
        )

    return loss, acc


def save_result(result, result_path):
    """Write JSON safely, replacing the previous version."""

    temp_path = result_path.with_suffix(".tmp")

    with open(temp_path, "w", encoding="utf-8") as file:
        json.dump(
            result,
            file,
            indent=4,
            ensure_ascii=False,
        )

    temp_path.replace(result_path)


def main():
    args = build_parser().parse_args()

    # 1. Load configuration.
    config_path = Path(args.config)

    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    seed = config["experiment"]["seed"]
    set_seed(seed)

    device = get_device(args.device)

    # 2. Prepare output directory.
    run_id = args.run_id or (
        datetime.now().strftime("%Y%m%d_%H%M%S")
        + f"_seed{seed}"
    )

    output_dir = Path(config["output"]["dir"]) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    result_path = output_dir / "result.json"

    print(f"Run ID: {run_id}")
    print(f"Device: {device}")

    # 3. Load Cora.
    data = load_cora(config["data"]["root"])

    x = data["x"].to(device)
    adj = data["adj"].to(device)
    y = data["y"].to(device)

    train_mask = data["train_mask"].to(device)
    val_mask = data["val_mask"].to(device)
    test_mask = data["test_mask"].to(device)

    adj_norm = normalize_adj(adj)

    num_features = x.shape[1]
    num_classes = int(y.max().item()) + 1

    print(f"Nodes: {x.shape[0]}")
    print(f"Features: {num_features}")
    print(f"Classes: {num_classes}")
    print(f"Train nodes: {train_mask.sum().item()}")
    print(f"Validation nodes: {val_mask.sum().item()}")
    print(f"Test nodes: {test_mask.sum().item()}")

    # 4. Build the model.
    model_config = config["model"]
    train_config = config["train"]

    model = GCN(
        in_dim=num_features,
        n_layers=model_config["num_layers"],
        hidden=model_config["hidden_dim"],
        num_classes=num_classes,
        dropout=model_config["dropout"],
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_config["learning_rate"],
        weight_decay=train_config["weight_decay"],
    )

    # 5. Prepare result structure.
    result = {
        "run_id": run_id,
        "dataset": config["data"]["name"],
        "model": "GCN",
        "num_layers": model_config["num_layers"],
        "hidden_dim": model_config["hidden_dim"],
        "dropout": model_config["dropout"],
        "seed": seed,
        "device": str(device),
        "epochs": train_config["epochs"],
        "history": [],
        "best_val_acc": 0.0,
        "best_epoch": 0,
        "test_acc": None,
        "status": "running",
    }

    best_val_acc = -1.0
    best_epoch = 0
    best_state = None

    # Save initial result before training.
    save_result(result, result_path)

    # 6. Training loop.
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

        # 7. Evaluate on validation set.
        val_loss, val_acc = evaluate(
            model,
            x,
            adj_norm,
            y,
            val_mask,
        )

        epoch_result = {
            "epoch": epoch,
            "train_loss": float(train_loss.item()),
            "train_acc": float(train_acc),
            "val_loss": float(val_loss),
            "val_acc": float(val_acc),
        }

        result["history"].append(epoch_result)

        # Keep the model with the best validation accuracy.
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch

            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

        result["best_val_acc"] = float(best_val_acc)
        result["best_epoch"] = best_epoch

        # Write every epoch to disk.
        save_result(result, result_path)

        print(
            f"Epoch {epoch:03d}/{train_config['epochs']} "
            f"Loss: {train_loss.item():.4f} "
            f"Train Acc: {train_acc:.4f} "
            f"Val Acc: {val_acc:.4f}"
        )

    # 8. Evaluate the best validation checkpoint on test set.
    if best_state is None:
        raise RuntimeError("Training did not produce a model checkpoint.")

    model.load_state_dict(best_state)

    test_loss, test_acc = evaluate(
        model,
        x,
        adj_norm,
        y,
        test_mask,
    )

    result["test_loss"] = float(test_loss)
    result["test_acc"] = float(test_acc)
    result["status"] = "completed"

    # 9. Save final result.
    save_result(result, result_path)

    print("\nTraining completed.")
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Result saved to: {result_path}")


if __name__ == "__main__":
    main()
