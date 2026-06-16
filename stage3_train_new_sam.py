import argparse
import math
import os
import sys
import time

# Ensure the repository root is importable when running as a standalone script.
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from train_18 import build_model, set_seed
from utils.dataset import get_cifar10_loaders
from optim.improved_sam import ImprovedSAM


def disable_running_stats(model):
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.backup_momentum = module.momentum
            module.momentum = 0


def enable_running_stats(model):
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            if hasattr(module, "backup_momentum"):
                module.momentum = module.backup_momentum


def get_dynamic_rho(epoch, total_epochs, rho_min=0.001, rho_max=0.05, warmup_epochs=5):
    """
    Dynamic rho schedule.

    Stage 1:
        Linear warmup from small rho to rho_max.

    Stage 2:
        Cosine decay from rho_max to rho_min.
    """

    if total_epochs <= 1:
        return rho_max

    if warmup_epochs <= 0:
        warmup_epochs = 1

    if epoch <= warmup_epochs:
        return rho_max * epoch / warmup_epochs

    progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)

    rho = rho_min + 0.5 * (rho_max - rho_min) * (1 + math.cos(math.pi * progress))

    return rho


def build_improved_sam_optimizer(
    model,
    lr,
    momentum,
    weight_decay,
    rho,
    adaptive=True,
    eta=0.01,
    exclude_bn_bias=True,
    adaptive_power=1.0
):
    optimizer = ImprovedSAM(
        model.parameters(),
        torch.optim.SGD,
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        rho=rho,
        adaptive=adaptive,
        eta=eta,
        exclude_bn_bias=exclude_bn_bias,
        adaptive_power=adaptive_power
    )

    return optimizer


def train_epoch(model, loader, criterion, optimizer, device, debug_batches=None):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0
    batch_count = 0

    for i, (images, labels) in enumerate(loader):
        if debug_batches is not None and i >= debug_batches:
            break

        batch_count += 1

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        # First forward-backward pass.
        # BN running stats are enabled.
        enable_running_stats(model)

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()

        optimizer.first_step(zero_grad=True)

        # Second forward-backward pass.
        # BN running stats should be disabled.
        disable_running_stats(model)

        outputs_second = model(images)
        loss_second = criterion(outputs_second, labels)
        loss_second.backward()

        optimizer.second_step(zero_grad=True)

        enable_running_stats(model)

        # Use first forward result as training metric.
        metric_outputs = outputs.detach()
        metric_loss = loss.detach()

        _, predicted = metric_outputs.max(1)

        total_loss += metric_loss.item() * images.size(0)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    train_loss = total_loss / total if total > 0 else 0.0
    train_acc = correct / total if total > 0 else 0.0

    return train_loss, train_acc, batch_count


def evaluate(model, loader, criterion, device, debug_batches=None):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0
    batch_count = 0

    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):
            if debug_batches is not None and i >= debug_batches:
                break

            batch_count += 1

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            _, predicted = outputs.max(1)

            total_loss += loss.item() * images.size(0)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    test_loss = total_loss / total if total > 0 else 0.0
    test_acc = correct / total if total > 0 else 0.0

    return test_loss, test_acc, batch_count


def load_previous_sam_history(prev_sam_log):
    if prev_sam_log is None:
        return None

    if not os.path.exists(prev_sam_log):
        print(f"Previous SAM log not found: {prev_sam_log}")
        return None

    prev_history = pd.read_csv(prev_sam_log).to_dict(orient="records")
    print(f"Loaded previous SAM log: {prev_sam_log}")

    return prev_history


def make_experiments(args):
    experiments = []

    if args.run_fixed_improved:
        for lr in args.sam_lrs:
            for rho in args.rhos:
                experiments.append({
                    "name": "fixed_improved_sam",
                    "lr": lr,
                    "rho": rho,
                    "rho_min": rho,
                    "rho_max": rho,
                    "warmup_epochs": 0,
                    "dynamic": False,
                    "label": f"Fixed ImprovedSAM lr={lr}, rho={rho}"
                })

    if args.run_dynamic_improved:
        for lr in args.dynamic_lrs:
            experiments.append({
                "name": "dynamic_improved_sam",
                "lr": lr,
                "rho": args.rho_max,
                "rho_min": args.rho_min,
                "rho_max": args.rho_max,
                "warmup_epochs": args.rho_warmup_epochs,
                "dynamic": True,
                "label": f"Dynamic ImprovedSAM lr={lr}, rho={args.rho_min}-{args.rho_max}"
            })

    return experiments


def run_one_experiment(args, exp, train_loader, test_loader, criterion, device):
    print(f"\nRunning experiment: {exp['label']}")

    model = build_model("resnet18", num_classes=10).to(device)

    optimizer = build_improved_sam_optimizer(
        model=model,
        lr=exp["lr"],
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        rho=exp["rho"],
        adaptive=True,
        eta=args.eta,
        exclude_bn_bias=True,
        adaptive_power=args.adaptive_power
    )

    history = []
    best_test_acc = 0.0
    best_epoch = 0
    best_model_path = ""

    rho_tag = (
        f"rho{exp['rho']}"
        if not exp["dynamic"]
        else f"rho{exp['rho_min']}-{exp['rho_max']}_warm{exp['warmup_epochs']}"
    )

    for epoch in range(1, args.epochs + 1):
        if exp["dynamic"]:
            current_rho = get_dynamic_rho(
                epoch=epoch,
                total_epochs=args.epochs,
                rho_min=exp["rho_min"],
                rho_max=exp["rho_max"],
                warmup_epochs=exp["warmup_epochs"]
            )
            optimizer.set_rho(current_rho)
        else:
            current_rho = exp["rho"]

        train_t0 = time.time()

        train_loss, train_acc, train_batches = train_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            debug_batches=(args.debug_batches if args.debug else None)
        )

        train_time = time.time() - train_t0

        eval_t0 = time.time()

        test_loss, test_acc, eval_batches = evaluate(
            model=model,
            loader=test_loader,
            criterion=criterion,
            device=device,
            debug_batches=(args.debug_batches if args.debug else None)
        )

        eval_time = time.time() - eval_t0
        epoch_time = train_time + eval_time

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch = epoch

            best_model_path = (
                f"results/models/stage3_resnet18_{exp['name']}_"
                f"lr{exp['lr']}_{rho_tag}_ep{args.epochs}_seed{args.seed}_best.pth"
            )

            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer": exp["name"],
                "lr": exp["lr"],
                "rho": current_rho,
                "rho_min": exp["rho_min"],
                "rho_max": exp["rho_max"],
                "dynamic": exp["dynamic"],
                "adaptive_power": args.adaptive_power,
                "eta": args.eta,
                "best_test_acc": best_test_acc,
                "seed": args.seed
            }, best_model_path)

        print(
            f"Epoch {epoch}/{args.epochs} | {exp['name']} | "
            f"LR: {exp['lr']} | Rho: {current_rho:.6f} | "
            f"Train Acc: {train_acc * 100:.2f}% | "
            f"Test Acc: {test_acc * 100:.2f}% | "
            f"Best: {best_test_acc * 100:.2f}% | "
            f"Train Batches: {train_batches} | Eval Batches: {eval_batches} | "
            f"Train Time: {train_time:.2f}s | Eval Time: {eval_time:.2f}s | "
            f"Total: {epoch_time:.2f}s"
        )

        record = {
            "epoch": epoch,
            "optimizer": exp["name"],
            "lr": exp["lr"],
            "rho": current_rho,
            "rho_min": exp["rho_min"],
            "rho_max": exp["rho_max"],
            "dynamic": exp["dynamic"],
            "adaptive_power": args.adaptive_power,
            "eta": args.eta,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "best_test_acc": best_test_acc,
            "epoch_time": epoch_time,
        }

        history.append(record)

    log_path = (
        f"results/logs/stage3_resnet18_{exp['name']}_"
        f"lr{exp['lr']}_{rho_tag}_ep{args.epochs}_seed{args.seed}.csv"
    )

    pd.DataFrame(history).to_csv(log_path, index=False)
    print(f"Saved log: {log_path}")

    summary_path = (
        f"results/tables/stage3_resnet18_{exp['name']}_"
        f"lr{exp['lr']}_{rho_tag}_ep{args.epochs}_seed{args.seed}_summary.csv"
    )

    pd.DataFrame([{
        "optimizer": exp["name"],
        "lr": exp["lr"],
        "rho": exp["rho"],
        "rho_min": exp["rho_min"],
        "rho_max": exp["rho_max"],
        "dynamic": exp["dynamic"],
        "adaptive_power": args.adaptive_power,
        "eta": args.eta,
        "label": exp["label"],
        "best_epoch": best_epoch,
        "best_test_acc": best_test_acc,
        "best_model_path": best_model_path,
        "seed": args.seed,
        "epochs": args.epochs
    }]).to_csv(summary_path, index=False)

    print(f"Saved summary: {summary_path}")

    return history


def plot_histories(histories, prev_sam_history, args):
    metrics = [
        ("train_loss", "Train Loss", False),
        ("test_loss", "Test Loss", False),
        ("train_acc", "Train Accuracy (%)", True),
        ("test_acc", "Test Accuracy (%)", True),
        ("rho", "Rho", False),
    ]

    for metric_key, ylabel, to_percent in metrics:
        plt.figure(figsize=(9, 6))

        for label, history in histories.items():
            epochs = [h["epoch"] for h in history]
            values = [h[metric_key] for h in history]

            if to_percent:
                values = [v * 100 for v in values]

            plt.plot(
                epochs,
                values,
                marker="o",
                linewidth=2,
                markersize=4,
                label=label
            )

        # Previous SAM is loaded only for performance metrics.
        # It usually has no dynamic rho column, so skip it for rho plot.
        if prev_sam_history is not None and metric_key != "rho":
            if metric_key in prev_sam_history[0]:
                epochs = [h["epoch"] for h in prev_sam_history]
                values = [h[metric_key] for h in prev_sam_history]

                if to_percent:
                    values = [v * 100 for v in values]

                plt.plot(
                    epochs,
                    values,
                    marker="x",
                    linewidth=2,
                    linestyle="--",
                    markersize=4,
                    label=args.prev_sam_label
                )

        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.title(f"ResNet18 CIFAR-10: Dynamic ImprovedSAM vs Previous SAM ({ylabel})")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        save_path = f"results/figures/stage3_dynamic_improved_sam_vs_prev_sam_{metric_key}.png"
        plt.savefig(save_path, dpi=300)
        plt.close()

        print(f"Saved figure: {save_path}")


def save_all_summary(histories, prev_sam_history, args):
    rows = []

    if prev_sam_history is not None:
        best_sam_acc = max(h["test_acc"] for h in prev_sam_history if "test_acc" in h)
        best_sam_epoch = max(
            prev_sam_history,
            key=lambda h: h["test_acc"] if "test_acc" in h else -1
        )["epoch"]

        rows.append({
            "method": args.prev_sam_label,
            "best_epoch": best_sam_epoch,
            "best_test_acc": best_sam_acc,
            "source": args.prev_sam_log
        })

    for label, history in histories.items():
        best_row = max(history, key=lambda h: h["test_acc"])
        rows.append({
            "method": label,
            "best_epoch": best_row["epoch"],
            "best_test_acc": best_row["test_acc"],
            "source": "current_run"
        })

    summary_path = "results/tables/stage3_dynamic_improved_sam_all_summary.csv"
    pd.DataFrame(rows).to_csv(summary_path, index=False)

    print(f"Saved all summary: {summary_path}")

    print("\n========== Best Test Accuracy Summary ==========")
    for row in rows:
        print(
            f"{row['method']} | "
            f"Best Epoch: {row['best_epoch']} | "
            f"Best Test Acc: {row['best_test_acc'] * 100:.2f}% | "
            f"Source: {row['source']}"
        )


def run_compare(args):
    os.makedirs("results/logs", exist_ok=True)
    os.makedirs("results/models", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)
    os.makedirs("results/tables", exist_ok=True)

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, test_loader = get_cifar10_loaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    criterion = nn.CrossEntropyLoss()

    prev_sam_history = load_previous_sam_history(args.prev_sam_log)

    experiments = make_experiments(args)

    if len(experiments) == 0:
        raise RuntimeError(
            "No experiment selected. Use --run_fixed_improved or --run_dynamic_improved."
        )

    histories = {}

    for exp in experiments:
        history = run_one_experiment(
            args=args,
            exp=exp,
            train_loader=train_loader,
            test_loader=test_loader,
            criterion=criterion,
            device=device
        )

        histories[exp["label"]] = history

    plot_histories(histories, prev_sam_history, args)
    save_all_summary(histories, prev_sam_history, args)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=2)

    parser.add_argument(
        "--run_fixed_improved",
        action="store_true",
        help="Run fixed-rho ImprovedSAM experiments."
    )

    parser.add_argument(
        "--run_dynamic_improved",
        action="store_true",
        help="Run dynamic-rho ImprovedSAM experiments."
    )

    parser.add_argument(
        "--sam_lrs",
        nargs="+",
        type=float,
        default=[0.01],
        help="Learning rates for fixed ImprovedSAM experiments."
    )

    parser.add_argument(
        "--rhos",
        nargs="+",
        type=float,
        default=[0.01, 0.05],
        help="Fixed rho values for fixed ImprovedSAM experiments."
    )

    parser.add_argument(
        "--dynamic_lrs",
        nargs="+",
        type=float,
        default=[0.01],
        help="Learning rates for Dynamic ImprovedSAM."
    )

    parser.add_argument(
        "--rho_min",
        type=float,
        default=0.001,
        help="Minimum rho for Dynamic ImprovedSAM."
    )

    parser.add_argument(
        "--rho_max",
        type=float,
        default=0.05,
        help="Maximum rho for Dynamic ImprovedSAM."
    )

    parser.add_argument(
        "--rho_warmup_epochs",
        type=int,
        default=5,
        help="Warmup epochs for dynamic rho."
    )

    parser.add_argument(
        "--eta",
        type=float,
        default=0.01,
        help="Eta for adaptive perturbation scale: s = |w| + eta."
    )

    parser.add_argument(
        "--adaptive_power",
        type=float,
        default=1.0,
        help="1.0 for mild adaptive SAM, 2.0 for ASAM-style perturbation."
    )

    parser.add_argument(
        "--prev_sam_log",
        type=str,
        default="results/logs/stage2_resnet18_cifar10_sam_lr0.01_ep100_seed42_rho0.1.csv",
        help="Existing SAM CSV log path. This script will load it for plotting and will not rerun SAM."
    )

    parser.add_argument(
        "--prev_sam_label",
        type=str,
        default="Previous Best SAM lr=0.01 rho=0.1",
        help="Label name for previous SAM curve."
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode with limited batches."
    )

    parser.add_argument(
        "--debug_batches",
        type=int,
        default=5,
        help="Number of batches per epoch in debug mode."
    )

    args = parser.parse_args()

    # Default behavior: only run Dynamic ImprovedSAM.
    # Previous SAM is loaded from CSV and will not be retrained.
    if not args.run_fixed_improved and not args.run_dynamic_improved:
        args.run_dynamic_improved = True

    return args


if __name__ == "__main__":
    args = parse_args()
    run_compare(args)