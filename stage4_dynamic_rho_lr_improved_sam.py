import argparse
import math
import os
import sys
import time

# Ensure repository root is importable.
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from train_18 import build_model, set_seed
from utils.dataset import get_cifar10_loaders
from optim.improved_sam2 import ImprovedSAM


# =========================
# Basic utilities
# =========================

def clamp(value, low, high):
    return max(low, min(high, value))


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


def set_optimizer_lr(optimizer, lr):
    """
    Set learning rate for ImprovedSAM.

    ImprovedSAM wraps a base_optimizer, so we update
    optimizer.base_optimizer.param_groups directly.
    """
    if lr < 0:
        raise ValueError(f"Invalid learning rate: {lr}")

    if hasattr(optimizer, "base_optimizer"):
        for group in optimizer.base_optimizer.param_groups:
            group["lr"] = lr
    else:
        for group in optimizer.param_groups:
            group["lr"] = lr


def get_optimizer_lr(optimizer):
    if hasattr(optimizer, "base_optimizer"):
        return optimizer.base_optimizer.param_groups[0]["lr"]
    return optimizer.param_groups[0]["lr"]


# =========================
# Dynamic hyperparameter schedules
# =========================

def get_dynamic_lr(
    epoch,
    total_epochs,
    lr_center=0.01,
    lr_min=1e-4,
    lr_max=0.012,
    warmup_epochs=5,
    oscillation_amp=0.02,
    oscillation_cycles=2
):
    """
    Dynamic learning rate schedule.

    Previous best SAM used:
        lr_center = 0.01

    Strategy:
        1. Warm up from lr_min to lr_center.
        2. Use cosine decay from lr_center to lr_min.
        3. Add small bounded oscillation.
        4. Clamp into [lr_min, lr_max].
    """

    if total_epochs <= 1:
        return lr_center

    warmup_epochs = max(1, warmup_epochs)

    if epoch <= warmup_epochs:
        lr = lr_min + (lr_center - lr_min) * epoch / warmup_epochs
        return clamp(lr, lr_min, lr_max)

    progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)

    # Main cosine decay: lr_center -> lr_min
    lr_decay = lr_min + 0.5 * (lr_center - lr_min) * (
        1 + math.cos(math.pi * progress)
    )

    # Small oscillation around the decay curve
    oscillation = oscillation_amp * lr_center * math.sin(
        2 * math.pi * oscillation_cycles * progress
    )

    lr = lr_decay + oscillation

    return clamp(lr, lr_min, lr_max)


def get_dynamic_rho(
    epoch,
    total_epochs,
    rho_center=0.1,
    rho_min=0.05,
    rho_max=0.12,
    warmup_epochs=5,
    oscillation_amp=0.03,
    oscillation_cycles=2
):
    """
    Dynamic rho schedule.

    Previous best SAM used:
        rho_center = 0.1

    Strategy:
        1. Warm up from rho_min to rho_center.
        2. Use cosine decay from rho_center to rho_min.
        3. Add small bounded oscillation.
        4. Clamp into [rho_min, rho_max].
    """

    if total_epochs <= 1:
        return rho_center

    warmup_epochs = max(1, warmup_epochs)

    if epoch <= warmup_epochs:
        rho = rho_min + (rho_center - rho_min) * epoch / warmup_epochs
        return clamp(rho, rho_min, rho_max)

    progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)

    # Main cosine decay: rho_center -> rho_min
    rho_decay = rho_min + 0.5 * (rho_center - rho_min) * (
        1 + math.cos(math.pi * progress)
    )

    # Small oscillation around the decay curve
    oscillation = oscillation_amp * rho_center * math.sin(
        2 * math.pi * oscillation_cycles * progress
    )

    rho = rho_decay + oscillation

    return clamp(rho, rho_min, rho_max)


# =========================
# Optimizer
# =========================

def build_dynamic_improved_sam_optimizer(
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


# =========================
# Train / Evaluate
# =========================

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
        # BN running stats are disabled.
        disable_running_stats(model)

        outputs_second = model(images)
        loss_second = criterion(outputs_second, labels)
        loss_second.backward()

        optimizer.second_step(zero_grad=True)

        enable_running_stats(model)

        # Training metric uses first forward result.
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


# =========================
# Load previous logs
# =========================

def load_history_csv(path, name):
    if path is None or path == "":
        return None

    if not os.path.exists(path):
        print(f"{name} log not found: {path}")
        return None

    history = pd.read_csv(path).to_dict(orient="records")
    print(f"Loaded {name} log: {path}")

    return history


# =========================
# Main experiment
# =========================

def run_experiment(args, train_loader, test_loader, criterion, device):
    adaptive = not args.no_adaptive

    method_name = (
        "stage4_dynamic_rho_lr_improved_sam"
        if adaptive
        else "stage4_dynamic_rho_lr_sam"
    )

    method_label = (
        f"Stage4 Dynamic Rho+LR ImprovedSAM "
        f"(lr_center={args.lr_center}, rho_center={args.rho_center})"
        if adaptive
        else
        f"Stage4 Dynamic Rho+LR SAM "
        f"(lr_center={args.lr_center}, rho_center={args.rho_center})"
    )

    exp_tag = (
        f"lrC{args.lr_center}_lrR{args.lr_min}-{args.lr_max}_"
        f"rhoC{args.rho_center}_rhoR{args.rho_min}-{args.rho_max}_"
        f"warm{args.warmup_epochs}_"
        f"lrOsc{args.lr_osc_amp}_rhoOsc{args.rho_osc_amp}_"
        f"ap{args.adaptive_power}_"
        f"seed{args.seed}"
    )

    print(f"\nRunning new Stage4 experiment: {method_label}")
    print(f"Experiment tag: {exp_tag}")

    model = build_model("resnet18", num_classes=10).to(device)

    optimizer = build_dynamic_improved_sam_optimizer(
        model=model,
        lr=args.lr_center,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        rho=args.rho_center,
        adaptive=adaptive,
        eta=args.eta,
        exclude_bn_bias=True,
        adaptive_power=args.adaptive_power
    )

    history = []
    best_test_acc = 0.0
    best_epoch = 0
    best_model_path = ""

    for epoch in range(1, args.epochs + 1):
        current_lr = get_dynamic_lr(
            epoch=epoch,
            total_epochs=args.epochs,
            lr_center=args.lr_center,
            lr_min=args.lr_min,
            lr_max=args.lr_max,
            warmup_epochs=args.warmup_epochs,
            oscillation_amp=args.lr_osc_amp,
            oscillation_cycles=args.lr_osc_cycles
        )

        current_rho = get_dynamic_rho(
            epoch=epoch,
            total_epochs=args.epochs,
            rho_center=args.rho_center,
            rho_min=args.rho_min,
            rho_max=args.rho_max,
            warmup_epochs=args.warmup_epochs,
            oscillation_amp=args.rho_osc_amp,
            oscillation_cycles=args.rho_osc_cycles
        )

        set_optimizer_lr(optimizer, current_lr)
        optimizer.set_rho(current_rho)

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
                f"results/models/{method_name}_{exp_tag}_"
                f"ep{args.epochs}_best.pth"
            )

            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "optimizer": method_name,
                "lr": current_lr,
                "rho": current_rho,
                "lr_center": args.lr_center,
                "lr_min": args.lr_min,
                "lr_max": args.lr_max,
                "rho_center": args.rho_center,
                "rho_min": args.rho_min,
                "rho_max": args.rho_max,
                "adaptive": adaptive,
                "adaptive_power": args.adaptive_power,
                "eta": args.eta,
                "best_test_acc": best_test_acc,
                "seed": args.seed,
                "epochs": args.epochs,
            }, best_model_path)

        print(
            f"Epoch {epoch}/{args.epochs} | {method_name} | "
            f"LR: {current_lr:.6f} | Rho: {current_rho:.6f} | "
            f"Train Acc: {train_acc * 100:.2f}% | "
            f"Test Acc: {test_acc * 100:.2f}% | "
            f"Best: {best_test_acc * 100:.2f}% | "
            f"Train Batches: {train_batches} | Eval Batches: {eval_batches} | "
            f"Train Time: {train_time:.2f}s | Eval Time: {eval_time:.2f}s | "
            f"Total: {epoch_time:.2f}s"
        )

        record = {
            "epoch": epoch,
            "optimizer": method_name,
            "lr": current_lr,
            "rho": current_rho,
            "lr_center": args.lr_center,
            "lr_min": args.lr_min,
            "lr_max": args.lr_max,
            "rho_center": args.rho_center,
            "rho_min": args.rho_min,
            "rho_max": args.rho_max,
            "adaptive": adaptive,
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
        f"results/logs/{method_name}_{exp_tag}_"
        f"ep{args.epochs}.csv"
    )

    pd.DataFrame(history).to_csv(log_path, index=False)
    print(f"Saved Stage4 log: {log_path}")

    summary_path = (
        f"results/tables/{method_name}_{exp_tag}_"
        f"ep{args.epochs}_summary.csv"
    )

    pd.DataFrame([{
        "optimizer": method_name,
        "label": method_label,
        "best_epoch": best_epoch,
        "best_test_acc": best_test_acc,
        "best_model_path": best_model_path,
        "log_path": log_path,
        "lr_center": args.lr_center,
        "lr_min": args.lr_min,
        "lr_max": args.lr_max,
        "rho_center": args.rho_center,
        "rho_min": args.rho_min,
        "rho_max": args.rho_max,
        "adaptive": adaptive,
        "adaptive_power": args.adaptive_power,
        "eta": args.eta,
        "seed": args.seed,
        "epochs": args.epochs
    }]).to_csv(summary_path, index=False)

    print(f"Saved Stage4 summary: {summary_path}")

    return method_name, method_label, history, log_path


# =========================
# Plotting
# =========================

def plot_metric_comparison(
    stage4_label,
    stage4_history,
    prev_sam_history,
    prev_dynamic_rho_history,
    args
):
    metrics = [
        ("train_loss", "Train Loss", False),
        ("test_loss", "Test Loss", False),
        ("train_acc", "Train Accuracy (%)", True),
        ("test_acc", "Test Accuracy (%)", True),
    ]

    for metric_key, ylabel, to_percent in metrics:
        plt.figure(figsize=(9, 6))

        epochs = [h["epoch"] for h in stage4_history]
        values = [h[metric_key] for h in stage4_history]

        if to_percent:
            values = [v * 100 for v in values]

        plt.plot(
            epochs,
            values,
            marker="o",
            linewidth=2,
            markersize=4,
            label=stage4_label
        )

        if prev_sam_history is not None and metric_key in prev_sam_history[0]:
            prev_epochs = [h["epoch"] for h in prev_sam_history]
            prev_values = [h[metric_key] for h in prev_sam_history]

            if to_percent:
                prev_values = [v * 100 for v in prev_values]

            plt.plot(
                prev_epochs,
                prev_values,
                marker="x",
                linewidth=2,
                linestyle="--",
                markersize=4,
                label=args.prev_sam_label
            )

        if prev_dynamic_rho_history is not None and metric_key in prev_dynamic_rho_history[0]:
            prev_epochs = [h["epoch"] for h in prev_dynamic_rho_history]
            prev_values = [h[metric_key] for h in prev_dynamic_rho_history]

            if to_percent:
                prev_values = [v * 100 for v in prev_values]

            plt.plot(
                prev_epochs,
                prev_values,
                marker="s",
                linewidth=2,
                linestyle=":",
                markersize=4,
                label=args.prev_dynamic_rho_label
            )

        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.title(f"Stage4 Dynamic Rho+LR ImprovedSAM Comparison ({ylabel})")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        save_path = (
            f"results/figures/stage4_dynamic_rho_lr_metric_{metric_key}.png"
        )
        plt.savefig(save_path, dpi=300)
        plt.close()

        print(f"Saved metric figure: {save_path}")


def plot_lr_curve(stage4_history):
    epochs = [h["epoch"] for h in stage4_history]
    lrs = [h["lr"] for h in stage4_history]

    plt.figure(figsize=(9, 6))
    plt.plot(
        epochs,
        lrs,
        marker="o",
        linewidth=2,
        markersize=4,
        label="Dynamic Learning Rate"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title("Stage4 Dynamic Learning Rate Schedule")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    save_path = "results/figures/stage4_dynamic_lr_curve.png"
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved LR curve: {save_path}")


def plot_rho_curve(stage4_history):
    epochs = [h["epoch"] for h in stage4_history]
    rhos = [h["rho"] for h in stage4_history]

    plt.figure(figsize=(9, 6))
    plt.plot(
        epochs,
        rhos,
        marker="o",
        linewidth=2,
        markersize=4,
        label="Dynamic Rho"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Rho")
    plt.title("Stage4 Dynamic Rho Schedule")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    save_path = "results/figures/stage4_dynamic_rho_curve.png"
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved Rho curve: {save_path}")


def save_all_summary(
    stage4_label,
    stage4_history,
    stage4_log_path,
    prev_sam_history,
    prev_dynamic_rho_history,
    args
):
    rows = []

    if prev_sam_history is not None:
        best_sam_row = max(prev_sam_history, key=lambda h: h["test_acc"])
        rows.append({
            "method": args.prev_sam_label,
            "best_epoch": best_sam_row["epoch"],
            "best_test_acc": best_sam_row["test_acc"],
            "source": args.prev_sam_log
        })

    if prev_dynamic_rho_history is not None:
        best_dynamic_rho_row = max(prev_dynamic_rho_history, key=lambda h: h["test_acc"])
        rows.append({
            "method": args.prev_dynamic_rho_label,
            "best_epoch": best_dynamic_rho_row["epoch"],
            "best_test_acc": best_dynamic_rho_row["test_acc"],
            "source": args.prev_dynamic_rho_log
        })

    best_stage4_row = max(stage4_history, key=lambda h: h["test_acc"])
    rows.append({
        "method": stage4_label,
        "best_epoch": best_stage4_row["epoch"],
        "best_test_acc": best_stage4_row["test_acc"],
        "source": stage4_log_path
    })

    summary_path = "results/tables/stage4_dynamic_rho_lr_all_summary.csv"
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


# =========================
# Run
# =========================

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

    prev_sam_history = load_history_csv(args.prev_sam_log, "previous SAM")
    prev_dynamic_rho_history = load_history_csv(
        args.prev_dynamic_rho_log,
        "previous Dynamic Rho ImprovedSAM"
    )

    method_name, stage4_label, stage4_history, stage4_log_path = run_experiment(
        args=args,
        train_loader=train_loader,
        test_loader=test_loader,
        criterion=criterion,
        device=device
    )

    plot_metric_comparison(
        stage4_label=stage4_label,
        stage4_history=stage4_history,
        prev_sam_history=prev_sam_history,
        prev_dynamic_rho_history=prev_dynamic_rho_history,
        args=args
    )

    # Two separate hyperparameter curves.
    plot_lr_curve(stage4_history)
    plot_rho_curve(stage4_history)

    save_all_summary(
        stage4_label=stage4_label,
        stage4_history=stage4_history,
        stage4_log_path=stage4_log_path,
        prev_sam_history=prev_sam_history,
        prev_dynamic_rho_history=prev_dynamic_rho_history,
        args=args
    )


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=2)

    # Center point from previous best SAM.
    parser.add_argument("--lr_center", type=float, default=0.01)
    parser.add_argument("--rho_center", type=float, default=0.1)

    # Dynamic LR range.
    parser.add_argument("--lr_min", type=float, default=1e-4)
    parser.add_argument("--lr_max", type=float, default=0.012)
    parser.add_argument("--lr_osc_amp", type=float, default=0.02)
    parser.add_argument("--lr_osc_cycles", type=int, default=2)

    # Dynamic rho range.
    parser.add_argument("--rho_min", type=float, default=0.05)
    parser.add_argument("--rho_max", type=float, default=0.12)
    parser.add_argument("--rho_osc_amp", type=float, default=0.03)
    parser.add_argument("--rho_osc_cycles", type=int, default=2)

    parser.add_argument("--warmup_epochs", type=int, default=5)

    parser.add_argument("--eta", type=float, default=0.01)
    parser.add_argument("--adaptive_power", type=float, default=1.0)

    parser.add_argument(
        "--no_adaptive",
        action="store_true",
        help="Disable adaptive perturbation. Then this becomes Dynamic Rho+LR SAM."
    )

    parser.add_argument(
        "--prev_sam_log",
        type=str,
        default="results/logs/stage2_resnet18_cifar10_sam_lr0.01_ep100_seed42_rho0.1.csv",
        help="Existing SAM CSV log path. This script loads it and does not rerun SAM."
    )

    parser.add_argument(
        "--prev_sam_label",
        type=str,
        default="Previous Best SAM lr=0.01 rho=0.1",
        help="Label for previous SAM curve."
    )

    parser.add_argument(
        "--prev_dynamic_rho_log",
        type=str,
        default="",
        help="Optional previous Dynamic Rho ImprovedSAM log path."
    )

    parser.add_argument(
        "--prev_dynamic_rho_label",
        type=str,
        default="Previous Dynamic Rho ImprovedSAM",
        help="Label for previous Dynamic Rho ImprovedSAM curve."
    )

    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug_batches", type=int, default=5)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_compare(args)