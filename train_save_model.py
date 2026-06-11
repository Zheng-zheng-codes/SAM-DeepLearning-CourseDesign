import argparse
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.optim import SGD, Adam, AdamW, RMSprop, Adagrad

from models.resnet import get_resnet18
from optim.sam import SAM
from utils.dataset import get_cifar10_loaders


# =========================
# 1. Random seed
# =========================

def set_seed(seed):
    """
    Fix random seed as much as possible.
    This makes different optimizer experiments more comparable.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Deterministic setting
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================
# 2. Optimizer builder
# =========================

def build_optimizer(optimizer_name, model, lr, momentum, weight_decay, rho):
    """
    Build optimizer.

    In this first-stage experiment, all optimizers use the same learning rate
    by default to satisfy the control-variable principle.

    Supported optimizers:
        - SGD
        - Adam
        - AdamW
        - RMSprop
        - Adagrad
        - SAM
    """

    if optimizer_name == "sgd":
        return SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay
        )

    if optimizer_name == "adam":
        return Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

    if optimizer_name == "adamw":
        return AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

    if optimizer_name == "rmsprop":
        return RMSprop(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay
        )

    if optimizer_name == "adagrad":
        return Adagrad(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

    if optimizer_name == "sam":
        return SAM(
            model.parameters(),
            base_optimizer=SGD,
            rho=rho,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay
        )

    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


# =========================
# 3. Train and evaluate
# =========================

def train_one_epoch(model, train_loader, criterion, optimizer, device, optimizer_name):
    """
    Train one epoch.

    For SAM:
        First step: move parameters to the adversarial neighborhood.
        Second step: compute gradient at that point and update original parameters.

    For fair metric recording:
        The training loss and accuracy are computed using the updated normal model
        parameters, not using SAM's perturbed loss.
    """

    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        if optimizer_name == "sam":
            # First forward-backward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.first_step(zero_grad=True)

            # Second forward-backward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.second_step(zero_grad=True)

            # Recompute metrics using the updated normal model parameters
            with torch.no_grad():
                metric_outputs = model(images)
                metric_loss = criterion(metric_outputs, labels)

        else:
            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Recompute metrics using the updated model parameters
            with torch.no_grad():
                metric_outputs = model(images)
                metric_loss = criterion(metric_outputs, labels)

        _, predicted = metric_outputs.max(1)

        total_loss += metric_loss.item() * images.size(0)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    train_loss = total_loss / total
    train_acc = correct / total

    return train_loss, train_acc


def evaluate(model, test_loader, criterion, device):
    """
    Evaluate model on test set.
    """

    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            _, predicted = outputs.max(1)

            total_loss += loss.item() * images.size(0)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    test_loss = total_loss / total
    test_acc = correct / total

    return test_loss, test_acc


# =========================
# 4. One experiment
# =========================

def run_one_experiment(optimizer_name, device, args):
    """
    Run one complete experiment for a specific optimizer.

    For fairness:
        1. Reset random seed before each optimizer.
        2. Rebuild dataloaders before each optimizer.
        3. Reinitialize model before each optimizer.
        4. Use the same lr, epochs, batch size, weight decay, and model.
    """

    print("\n" + "=" * 80)
    print(f"Start experiment: {optimizer_name.upper()}")
    print("=" * 80)

    set_seed(args.seed)

    train_loader, test_loader = get_cifar10_loaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    model = get_resnet18(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = build_optimizer(
        optimizer_name=optimizer_name,
        model=model,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        rho=args.rho
    )

    log_name = (
        f"stage1_resnet18_cifar10_{optimizer_name}"
        f"_lr{args.lr}"
        f"_ep{args.epochs}"
        f"_seed{args.seed}"
    )

    if optimizer_name == "sam":
        log_name += f"_rho{args.rho}"

    best_model_path = f"results/models/{log_name}_best.pth"

    history = []
    best_test_acc = 0.0

    total_start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()

        train_loss, train_acc = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            optimizer_name=optimizer_name
        )

        test_loss, test_acc = evaluate(
            model=model,
            test_loader=test_loader,
            criterion=criterion,
            device=device
        )

        epoch_time = time.time() - epoch_start_time

        if test_acc > best_test_acc:
            best_test_acc = test_acc

            torch.save(
                {
                    "epoch": epoch,
                    "stage": "stage1",
                    "model": "resnet18",
                    "dataset": "cifar10",
                    "optimizer": optimizer_name,
                    "model_state_dict": model.state_dict(),
                    "lr": args.lr,
                    "momentum": args.momentum if optimizer_name in ["sgd", "rmsprop", "sam"] else None,
                    "weight_decay": args.weight_decay,
                    "rho": args.rho if optimizer_name == "sam" else None,
                    "best_test_acc": best_test_acc,
                    "seed": args.seed
                },
                best_model_path
            )

            print(
                f"Saved best model: {best_model_path} "
                f"(Epoch {epoch}, Best Acc: {best_test_acc * 100:.2f}%)"
            )

        record = {
            "epoch": epoch,
            "stage": "stage1",
            "optimizer": optimizer_name,
            "lr": args.lr,
            "momentum": args.momentum if optimizer_name in ["sgd", "rmsprop", "sam"] else None,
            "weight_decay": args.weight_decay,
            "rho": args.rho if optimizer_name == "sam" else None,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "best_test_acc": best_test_acc,
            "epoch_time": epoch_time
        }

        history.append(record)

        print(
            f"Optimizer: {optimizer_name.upper()} | "
            f"Epoch [{epoch:03d}/{args.epochs:03d}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc * 100:.2f}% | "
            f"Test Loss: {test_loss:.4f} | "
            f"Test Acc: {test_acc * 100:.2f}% | "
            f"Best Acc: {best_test_acc * 100:.2f}% | "
            f"Time: {epoch_time:.2f}s"
        )

    total_time = time.time() - total_start_time

    log_path = f"results/logs/{log_name}.csv"
    pd.DataFrame(history).to_csv(log_path, index=False)

    summary = {
        "stage": "stage1",
        "model": "resnet18",
        "dataset": "cifar10",
        "optimizer": optimizer_name,
        "lr": args.lr,
        "momentum": args.momentum if optimizer_name in ["sgd", "rmsprop", "sam"] else None,
        "weight_decay": args.weight_decay,
        "rho": args.rho if optimizer_name == "sam" else None,
        "epochs": args.epochs,
        "seed": args.seed,
        "best_test_acc": best_test_acc,
        "final_test_acc": history[-1]["test_acc"],
        "total_time": total_time,
        "best_model_path": best_model_path
    }

    summary_path = f"results/tables/{log_name}_summary.csv"
    pd.DataFrame([summary]).to_csv(summary_path, index=False)

    print(f"\nFinished {optimizer_name.upper()} experiment.")
    print(f"Log saved to: {log_path}")
    print(f"Summary saved to: {summary_path}")
    print(f"Best model saved to: {best_model_path}")
    print(f"Best test accuracy: {best_test_acc * 100:.2f}%")
    print(f"Final test accuracy: {history[-1]['test_acc'] * 100:.2f}%")
    print(f"Total training time: {total_time:.2f}s")

    return history, summary


# =========================
# 5. Plot and summary
# =========================

def plot_metric(all_histories, metric, ylabel, title, save_path):
    """
    Plot comparison curve.
    """

    plt.figure(figsize=(8, 6))

    for optimizer_name, history in all_histories.items():
        epochs = [item["epoch"] for item in history]
        values = [item[metric] for item in history]

        if "acc" in metric:
            values = [v * 100 for v in values]

        plt.plot(
            epochs,
            values,
            marker="o",
            linewidth=2,
            label=optimizer_name.upper()
        )

    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Figure saved to: {save_path}")


def save_total_summary(all_summaries, args):
    """
    Save summary table for all optimizers.
    """

    summary_path = (
        f"results/tables/stage1_optimizer_comparison"
        f"_lr{args.lr}"
        f"_ep{args.epochs}"
        f"_seed{args.seed}.csv"
    )

    df = pd.DataFrame(all_summaries)
    df.to_csv(summary_path, index=False)

    print(f"\nTotal summary saved to: {summary_path}")
    print("\nStage1 optimizer comparison summary:")
    print(df)


# =========================
# 6. Main
# =========================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--optimizers",
        nargs="+",
        default=["sgd", "adam", "adamw", "rmsprop", "adagrad", "sam"],
        choices=["sgd", "adam", "adamw", "rmsprop", "adagrad", "sam"],
        help="Optimizers to compare."
    )

    # Default setting for the expanded first-stage comparison
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.01)

    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--rho", type=float, default=0.05)

    parser.add_argument("--seed", type=int, default=42)

    # With GPU, you can set num_workers to 2 or 4 for faster data loading.
    # If there is any dataloader problem, set it back to 0.
    parser.add_argument("--num_workers", type=int, default=2)

    args = parser.parse_args()

    os.makedirs("results/logs", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)
    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("results/models", exist_ok=True)

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("Expanded first-stage optimizer comparison experiment")
    print("=" * 80)
    print(f"Using device: {device}")
    print(f"Optimizers: {args.optimizers}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Weight decay: {args.weight_decay}")
    print(f"Momentum for SGD/RMSprop/SAM: {args.momentum}")
    print(f"Rho for SAM: {args.rho}")
    print(f"Seed: {args.seed}")
    print(f"Num workers: {args.num_workers}")
    print("=" * 80)

    all_histories = {}
    all_summaries = []

    for optimizer_name in args.optimizers:
        history, summary = run_one_experiment(
            optimizer_name=optimizer_name,
            device=device,
            args=args
        )

        all_histories[optimizer_name] = history
        all_summaries.append(summary)

    save_total_summary(all_summaries, args)

    fig_prefix = f"stage1_lr{args.lr}_ep{args.epochs}_seed{args.seed}"

    plot_metric(
        all_histories=all_histories,
        metric="train_loss",
        ylabel="Train Loss",
        title="Stage1 Training Loss Comparison",
        save_path=f"results/figures/train_loss_comparison_{fig_prefix}.png"
    )

    plot_metric(
        all_histories=all_histories,
        metric="test_loss",
        ylabel="Test Loss",
        title="Stage1 Test Loss Comparison",
        save_path=f"results/figures/test_loss_comparison_{fig_prefix}.png"
    )

    plot_metric(
        all_histories=all_histories,
        metric="train_acc",
        ylabel="Train Accuracy (%)",
        title="Stage1 Training Accuracy Comparison",
        save_path=f"results/figures/train_accuracy_comparison_{fig_prefix}.png"
    )

    plot_metric(
        all_histories=all_histories,
        metric="test_acc",
        ylabel="Test Accuracy (%)",
        title="Stage1 Test Accuracy Comparison",
        save_path=f"results/figures/test_accuracy_comparison_{fig_prefix}.png"
    )

    print("\nAll expanded first-stage optimizer comparison experiments finished.")


if __name__ == "__main__":
    main()