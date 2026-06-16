import argparse
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.optim import SGD, Adam, AdamW, RMSprop, Adagrad

from models.resnet import get_resnet18
from optim.sam import SAM
from optim.improved_sam import ImprovedSAM
from utils.dataset import get_cifar10_loaders


# =========================
# 1. Random seed
# =========================

def set_seed(seed):
    """
    Fix random seed as much as possible.
    This makes different experiments more comparable.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================
# 2. Model builder
# =========================

def build_model(model_name, num_classes=10):
    """
    Build model by name.
    """
    if model_name == "resnet18":
        return get_resnet18(num_classes=num_classes)

    raise ValueError(f"Unsupported model: {model_name}")


# =========================
# 3. Optimizer builder
# =========================

def build_optimizer(optimizer_name, model, lr, momentum, weight_decay, rho):
    """
    Build optimizer for stage2 parameter tuning.
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

    if optimizer_name == "improved_sam":
        return ImprovedSAM(
            model.parameters(),
            base_optimizer=SGD,
            rho=rho,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay
        )

    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


# =========================
# 4. Train and evaluate
# =========================

def train_one_epoch(model, train_loader, criterion, optimizer, device, optimizer_name):
    """
    Train one epoch.

    For SAM, one update contains two forward-backward passes.
    For fair metric recording, loss and accuracy are recomputed
    using the updated normal model parameters.
    """

    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        if optimizer_name == "sam":
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.first_step(zero_grad=True)

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.second_step(zero_grad=True)

            with torch.no_grad():
                metric_outputs = model(images)
                metric_loss = criterion(metric_outputs, labels)

        else:
            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

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
# 5. Plot single experiment
# =========================

def plot_single_experiment(history, save_prefix):
    """
    Save four curves for one parameter setting:
        train loss
        test loss
        train accuracy
        test accuracy
    """

    epochs = [item["epoch"] for item in history]

    train_loss = [item["train_loss"] for item in history]
    test_loss = [item["test_loss"] for item in history]
    train_acc = [item["train_acc"] * 100 for item in history]
    test_acc = [item["test_acc"] * 100 for item in history]

    # Train loss
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_loss, marker='o', linewidth=2, markersize=4)
    plt.xlabel("Epoch")
    plt.ylabel("Train Loss")
    plt.title("Train Loss Curve")
    plt.grid(True)
    plt.tight_layout()
    path = f"{save_prefix}_train_loss.png"
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Figure saved to: {path}")

    # Test loss
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, test_loss, marker='o', linewidth=2, markersize=4)
    plt.xlabel("Epoch")
    plt.ylabel("Test Loss")
    plt.title("Test Loss Curve")
    plt.grid(True)
    plt.tight_layout()
    path = f"{save_prefix}_test_loss.png"
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Figure saved to: {path}")

    # Train accuracy
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_acc, marker='o', linewidth=2, markersize=4)
    plt.xlabel("Epoch")
    plt.ylabel("Train Accuracy (%)")
    plt.title("Train Accuracy Curve")
    plt.grid(True)
    plt.tight_layout()
    path = f"{save_prefix}_train_acc.png"
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Figure saved to: {path}")

    # Test accuracy
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, test_acc, marker='o', linewidth=2, markersize=4)
    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy (%)")
    plt.title("Test Accuracy Curve")
    plt.grid(True)
    plt.tight_layout()
    path = f"{save_prefix}_test_acc.png"
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Figure saved to: {path}")


# =========================
# 6. Plot parameter comparison
# =========================

def plot_parameter_comparison(group_histories, metric, ylabel, title, save_path):
    """
    Plot comparison curves for the same model and optimizer
    under different parameter settings.
    """

    plt.figure(figsize=(9, 6))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for idx, (label, history) in enumerate(group_histories.items()):
        epochs = [item["epoch"] for item in history]
        values = [item[metric] for item in history]

        if "acc" in metric:
            values = [v * 100 for v in values]

        plt.plot(
            epochs,
            values,
            marker='o',
            linewidth=2,
            markersize=4,
            color=colors[idx % len(colors)],
            label=label
        )

    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Comparison figure saved to: {save_path}")


def save_group_comparison_figures(all_histories):
    """
    For each model and optimizer, compare different parameter settings.
    """

    os.makedirs("results/figures/stage2_comparison", exist_ok=True)

    grouped = {}

    for item in all_histories:
        model_name = item["model"]
        optimizer_name = item["optimizer"]
        label = item["label"]
        history = item["history"]

        group_key = (model_name, optimizer_name)

        if group_key not in grouped:
            grouped[group_key] = {}

        grouped[group_key][label] = history

    for (model_name, optimizer_name), group_histories in grouped.items():
        prefix = f"results/figures/stage2_comparison/stage2_{model_name}_{optimizer_name}"

        plot_parameter_comparison(
            group_histories=group_histories,
            metric="test_acc",
            ylabel="Test Accuracy (%)",
            title=f"{model_name.upper()} {optimizer_name.upper()} Test Accuracy under Different Parameters",
            save_path=f"{prefix}_test_acc_comparison.png"
        )

        plot_parameter_comparison(
            group_histories=group_histories,
            metric="test_loss",
            ylabel="Test Loss",
            title=f"{model_name.upper()} {optimizer_name.upper()} Test Loss under Different Parameters",
            save_path=f"{prefix}_test_loss_comparison.png"
        )

        plot_parameter_comparison(
            group_histories=group_histories,
            metric="train_acc",
            ylabel="Train Accuracy (%)",
            title=f"{model_name.upper()} {optimizer_name.upper()} Train Accuracy under Different Parameters",
            save_path=f"{prefix}_train_acc_comparison.png"
        )

        plot_parameter_comparison(
            group_histories=group_histories,
            metric="train_loss",
            ylabel="Train Loss",
            title=f"{model_name.upper()} {optimizer_name.upper()} Train Loss under Different Parameters",
            save_path=f"{prefix}_train_loss_comparison.png"
        )


# =========================
# 7. Run one experiment
# =========================

def run_one_experiment(model_name, optimizer_name, lr, rho, device, args):
    """
    Run one complete experiment for one model, one optimizer, and one parameter setting.
    """

    print("\n" + "=" * 80)
    print(
        f"Stage2 experiment | "
        f"Model: {model_name} | "
        f"Optimizer: {optimizer_name} | "
        f"LR: {lr} | "
        f"Rho: {rho if optimizer_name == 'sam' else 'None'}"
    )
    print("=" * 80)

    set_seed(args.seed)

    train_loader, test_loader = get_cifar10_loaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    model = build_model(model_name=model_name, num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = build_optimizer(
        optimizer_name=optimizer_name,
        model=model,
        lr=lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        rho=rho
    )

    log_name = (
        f"stage2_{model_name}_cifar10_{optimizer_name}"
        f"_lr{lr}"
        f"_ep{args.epochs}"
        f"_seed{args.seed}"
    )

    if optimizer_name == "sam":
        log_name += f"_rho{rho}"

    log_path = f"results/logs/{log_name}.csv"
    summary_path = f"results/tables/{log_name}_summary.csv"
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
                    "stage": "stage2",
                    "model": model_name,
                    "dataset": "cifar10",
                    "optimizer": optimizer_name,
                    "model_state_dict": model.state_dict(),
                    "lr": lr,
                    "momentum": args.momentum if optimizer_name in ["sgd", "rmsprop", "sam"] else None,
                    "weight_decay": args.weight_decay,
                    "rho": rho if optimizer_name == "sam" else None,
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
            "stage": "stage2",
            "model": model_name,
            "optimizer": optimizer_name,
            "lr": lr,
            "momentum": args.momentum if optimizer_name in ["sgd", "rmsprop", "sam"] else None,
            "weight_decay": args.weight_decay,
            "rho": rho if optimizer_name == "sam" else None,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "best_test_acc": best_test_acc,
            "epoch_time": epoch_time
        }

        history.append(record)

        print(
            f"Model: {model_name.upper()} | "
            f"Optimizer: {optimizer_name.upper()} | "
            f"LR: {lr} | "
            f"Rho: {rho if optimizer_name == 'sam' else 'None'} | "
            f"Epoch [{epoch:03d}/{args.epochs:03d}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc * 100:.2f}% | "
            f"Test Loss: {test_loss:.4f} | "
            f"Test Acc: {test_acc * 100:.2f}% | "
            f"Best Acc: {best_test_acc * 100:.2f}% | "
            f"Time: {epoch_time:.2f}s"
        )

    total_time = time.time() - total_start_time

    pd.DataFrame(history).to_csv(log_path, index=False)

    summary = {
        "stage": "stage2",
        "model": model_name,
        "dataset": "cifar10",
        "optimizer": optimizer_name,
        "lr": lr,
        "momentum": args.momentum if optimizer_name in ["sgd", "rmsprop", "sam"] else None,
        "weight_decay": args.weight_decay,
        "rho": rho if optimizer_name == "sam" else None,
        "epochs": args.epochs,
        "seed": args.seed,
        "best_test_acc": best_test_acc,
        "final_test_acc": history[-1]["test_acc"],
        "total_time": total_time,
        "best_model_path": best_model_path,
        "log_path": log_path
    }

    pd.DataFrame([summary]).to_csv(summary_path, index=False)

    # Single-experiment plots disabled. Comparison plots are generated later.

    print(f"\nFinished stage2 experiment.")
    print(f"Log saved to: {log_path}")
    print(f"Summary saved to: {summary_path}")
    print(f"Best model saved to: {best_model_path}")
    print(f"Best test accuracy: {best_test_acc * 100:.2f}%")
    print(f"Final test accuracy: {history[-1]['test_acc'] * 100:.2f}%")
    print(f"Total training time: {total_time:.2f}s")

    return history, summary


# =========================
# 8. Build experiment list
# =========================

def build_experiment_list(args):
    """
    Build parameter tuning experiments.

    Non-SAM optimizers:
        tune learning rate.

    SAM:
        tune learning rate and rho.
    """

    experiments = []

    for model_name in args.models:
        for optimizer_name in args.optimizers:

            if optimizer_name == "sam":
                for lr in args.sam_lrs:
                    for rho in args.rhos:
                        experiments.append(
                            {
                                "model": model_name,
                                "optimizer": optimizer_name,
                                "lr": lr,
                                "rho": rho,
                                "label": f"lr={lr}, rho={rho}"
                            }
                        )
            else:
                for lr in args.lrs:
                    experiments.append(
                        {
                            "model": model_name,
                            "optimizer": optimizer_name,
                            "lr": lr,
                            "rho": None,
                            "label": f"lr={lr}"
                        }
                    )

    return experiments


# =========================
# 9. Main
# =========================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--models",
        nargs="+",
        default=["resnet18"],
        choices=["resnet18"],
        help="Model(s) to tune. Only ResNet18 is supported in this script."
    )

    parser.add_argument(
        "--optimizers",
        nargs="+",
        default=["sgd", "adam", "adamw", "rmsprop", "adagrad", "sam"],
        choices=["sgd", "adam", "adamw", "rmsprop", "adagrad", "sam"],
        help="Optimizers to tune."
    )

    # Learning rates for ordinary optimizers
    parser.add_argument(
        "--lrs",
        nargs="+",
        type=float,
        default=[0.1, 0.01, 0.001],
        help="Learning rates for non-SAM optimizers."
    )

    # Learning rates for SAM
    parser.add_argument(
        "--sam_lrs",
        nargs="+",
        type=float,
        default=[0.1, 0.01, 0.001],
        help="Learning rates for SAM."
    )

    parser.add_argument(
        "--rhos",
        nargs="+",
        type=float,
        default=[0.01, 0.05, 0.1],
        help="Rho values for SAM."
    )

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)

    # With GPU, 2 or 4 is usually fine.
    # If dataloader has problems, set it to 0.
    parser.add_argument("--num_workers", type=int, default=2)

    args = parser.parse_args()

    os.makedirs("results/logs", exist_ok=True)
    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("results/models", exist_ok=True)
    os.makedirs("results/figures/stage2_comparison", exist_ok=True)

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("Stage2 parameter tuning experiment")
    print("=" * 80)
    print(f"Using device: {device}")
    print(f"Models: {args.models}")
    print(f"Optimizers: {args.optimizers}")
    print(f"Non-SAM learning rates: {args.lrs}")
    print(f"SAM learning rates: {args.sam_lrs}")
    print(f"SAM rhos: {args.rhos}")
    print(f"Epochs per experiment: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Momentum: {args.momentum}")
    print(f"Weight decay: {args.weight_decay}")
    print(f"Seed: {args.seed}")
    print(f"Num workers: {args.num_workers}")
    print("=" * 80)

    experiments = build_experiment_list(args)

    print(f"Total experiments: {len(experiments)}")
    print("=" * 80)

    # Group experiments by (model, optimizer) so we can run
    # and save comparison plots per optimizer immediately.
    groups = {}
    for exp in experiments:
        key = (exp["model"], exp["optimizer"])
        groups.setdefault(key, []).append(exp)

    all_summaries = []

    for (model_name, optimizer_name), group in groups.items():
        print(f"\nRunning group: Model={model_name}, Optimizer={optimizer_name} ({len(group)} runs)")

        group_items = []

        for exp in group:
            history, summary = run_one_experiment(
                model_name=exp["model"],
                optimizer_name=exp["optimizer"],
                lr=exp["lr"],
                rho=exp["rho"],
                device=device,
                args=args
            )

            item = {
                "model": exp["model"],
                "optimizer": exp["optimizer"],
                "lr": exp["lr"],
                "rho": exp["rho"],
                "label": exp["label"],
                "history": history
            }

            group_items.append(item)
            all_summaries.append(summary)

        # Build dict expected by plot_parameter_comparison: label -> history
        group_histories = {it["label"]: it["history"] for it in group_items}

        prefix = f"results/figures/stage2_comparison/stage2_{model_name}_{optimizer_name}"

        plot_parameter_comparison(
            group_histories=group_histories,
            metric="test_acc",
            ylabel="Test Accuracy (%)",
            title=f"{model_name.upper()} {optimizer_name.upper()} Test Accuracy under Different Parameters",
            save_path=f"{prefix}_test_acc_comparison.png"
        )

        plot_parameter_comparison(
            group_histories=group_histories,
            metric="test_loss",
            ylabel="Test Loss",
            title=f"{model_name.upper()} {optimizer_name.upper()} Test Loss under Different Parameters",
            save_path=f"{prefix}_test_loss_comparison.png"
        )

        plot_parameter_comparison(
            group_histories=group_histories,
            metric="train_acc",
            ylabel="Train Accuracy (%)",
            title=f"{model_name.upper()} {optimizer_name.upper()} Train Accuracy under Different Parameters",
            save_path=f"{prefix}_train_acc_comparison.png"
        )

        plot_parameter_comparison(
            group_histories=group_histories,
            metric="train_loss",
            ylabel="Train Loss",
            title=f"{model_name.upper()} {optimizer_name.upper()} Train Loss under Different Parameters",
            save_path=f"{prefix}_train_loss_comparison.png"
        )

        print(f"Saved comparison figures for {model_name} {optimizer_name} to {prefix}_*.png")

    summary_all_path = (
        f"results/tables/stage2_tuning_summary_all"
        f"_ep{args.epochs}"
        f"_seed{args.seed}.csv"
    )

    pd.DataFrame(all_summaries).to_csv(summary_all_path, index=False)

    print(f"\nAll stage2 summaries saved to: {summary_all_path}")

    print("\nAll stage2 parameter tuning experiments finished.")


if __name__ == "__main__":
    main()