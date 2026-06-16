import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from train_18 import build_model, set_seed
from utils.dataset import get_cifar10_loaders
from optim.improved_sam3 import ImprovedSAM3


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


def make_safe_tag(tag):
    """
    Make a short and safe filename tag.
    Avoid too long filenames and avoid dots in filenames.
    """
    tag = tag.replace(".", "p")
    tag = tag.replace("/", "_")
    tag = tag.replace(" ", "_")
    return tag


# =========================
# Mild feedback controller with recovery
# =========================

class MildRecoverGapAwareFeedbackController:
    """
    Mild gap-aware feedback controller with recovery.

    Mechanism 1:
        If test loss does not improve for several epochs,
        reduce lr and rho mildly.

    Mechanism 2:
        If train_acc - test_acc is too large for several epochs,
        reduce lr and rho mildly.

    Mechanism 3:
        If loss improves and gap is safe for several epochs,
        recover lr and rho mildly.

    Mechanism 4:
        Cooldown avoids repeated adjustments in consecutive epochs.
    """

    def __init__(
        self,
        lr_init=0.01,
        rho_init=0.1,
        lr_min=0.001,
        lr_max=0.012,
        rho_min=0.05,
        rho_max=0.12,
        patience=8,
        min_delta=1e-4,
        lr_factor=0.8,
        rho_factor=0.9,
        gap_threshold=0.05,
        gap_lr_factor=0.9,
        gap_rho_factor=0.95,
        gap_patience=4,
        cooldown=5,
        recover_patience=5,
        recover_lr_factor=1.03,
        recover_rho_factor=1.02,
        recover_gap_threshold=0.03,
    ):
        self.lr_init = lr_init
        self.rho_init = rho_init

        self.lr = clamp(lr_init, lr_min, lr_max)
        self.rho = clamp(rho_init, rho_min, rho_max)

        self.lr_min = lr_min
        self.lr_max = lr_max
        self.rho_min = rho_min
        self.rho_max = rho_max

        self.patience = patience
        self.min_delta = min_delta
        self.lr_factor = lr_factor
        self.rho_factor = rho_factor

        self.gap_threshold = gap_threshold
        self.gap_lr_factor = gap_lr_factor
        self.gap_rho_factor = gap_rho_factor
        self.gap_patience = gap_patience

        self.cooldown = cooldown
        self.cooldown_counter = 0

        self.recover_patience = recover_patience
        self.recover_lr_factor = recover_lr_factor
        self.recover_rho_factor = recover_rho_factor
        self.recover_gap_threshold = recover_gap_threshold

        self.best_loss = float("inf")
        self.bad_epochs = 0
        self.bad_gap_epochs = 0
        self.good_epochs = 0

        self.adjust_count = 0
        self.recover_count = 0

    def _apply_decay(self, lr_factor, rho_factor):
        old_lr = self.lr
        old_rho = self.rho

        self.lr = clamp(self.lr * lr_factor, self.lr_min, self.lr_max)
        self.rho = clamp(self.rho * rho_factor, self.rho_min, self.rho_max)

        changed = (
            abs(self.lr - old_lr) > 1e-12
            or abs(self.rho - old_rho) > 1e-12
        )

        return changed

    def _apply_recovery(self):
        old_lr = self.lr
        old_rho = self.rho

        # Recover only up to initial values.
        self.lr = min(self.lr * self.recover_lr_factor, self.lr_init)
        self.rho = min(self.rho * self.recover_rho_factor, self.rho_init)

        self.lr = clamp(self.lr, self.lr_min, self.lr_max)
        self.rho = clamp(self.rho, self.rho_min, self.rho_max)

        changed = (
            abs(self.lr - old_lr) > 1e-12
            or abs(self.rho - old_rho) > 1e-12
        )

        return changed

    def step(self, train_acc, val_acc, val_loss):
        adjusted = False
        recovered = False
        reasons = []

        gap = train_acc - val_acc
        in_cooldown = self.cooldown_counter > 0

        # Rule 1: loss improvement / plateau
        loss_improved = False

        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.bad_epochs = 0
            loss_improved = True
        else:
            self.bad_epochs += 1

        # Rule 2: generalization gap
        if gap > self.gap_threshold:
            self.bad_gap_epochs += 1
        else:
            self.bad_gap_epochs = 0

        # Rule 3: recovery condition
        gap_safe_for_recovery = gap < self.recover_gap_threshold

        if loss_improved and gap_safe_for_recovery:
            self.good_epochs += 1
        else:
            self.good_epochs = 0

        loss_ready = self.bad_epochs >= self.patience
        gap_ready = self.bad_gap_epochs >= self.gap_patience
        recover_ready = self.good_epochs >= self.recover_patience

        # Priority:
        # 1. Decay if risk exists.
        # 2. Recover if training is healthy.
        # 3. Do nothing during cooldown.
        if (not in_cooldown) and (loss_ready or gap_ready):
            if loss_ready and gap_ready:
                use_lr_factor = min(self.lr_factor, self.gap_lr_factor)
                use_rho_factor = min(self.rho_factor, self.gap_rho_factor)
                reasons.append("loss_plateau+generalization_gap")
            elif loss_ready:
                use_lr_factor = self.lr_factor
                use_rho_factor = self.rho_factor
                reasons.append("loss_plateau")
            else:
                use_lr_factor = self.gap_lr_factor
                use_rho_factor = self.gap_rho_factor
                reasons.append("generalization_gap")

            changed = self._apply_decay(use_lr_factor, use_rho_factor)

            if changed:
                adjusted = True
                self.adjust_count += 1
                self.cooldown_counter = self.cooldown

            if loss_ready:
                self.bad_epochs = 0
            if gap_ready:
                self.bad_gap_epochs = 0

            self.good_epochs = 0

        elif (not in_cooldown) and recover_ready:
            changed = self._apply_recovery()

            if changed:
                adjusted = True
                recovered = True
                self.adjust_count += 1
                self.recover_count += 1
                self.cooldown_counter = self.cooldown
                reasons.append("recovery")

            self.good_epochs = 0

        else:
            if self.cooldown_counter > 0:
                self.cooldown_counter -= 1

        if len(reasons) == 0:
            reason = "none"
        else:
            reason = "+".join(reasons)

        return {
            "lr": self.lr,
            "rho": self.rho,
            "gap": gap,
            "adjusted": adjusted,
            "recovered": recovered,
            "reason": reason,
            "best_loss": self.best_loss,
            "bad_epochs": self.bad_epochs,
            "bad_gap_epochs": self.bad_gap_epochs,
            "good_epochs": self.good_epochs,
            "adjust_count": self.adjust_count,
            "recover_count": self.recover_count,
            "cooldown_counter": self.cooldown_counter,
        }


# =========================
# Optimizer builder
# =========================

def build_feedback_sam_optimizer(
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
    optimizer = ImprovedSAM3(
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
        enable_running_stats(model)

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()

        optimizer.first_step(zero_grad=True)

        # Second forward-backward pass.
        disable_running_stats(model)

        outputs_second = model(images)
        loss_second = criterion(outputs_second, labels)
        loss_second.backward()

        optimizer.second_step(zero_grad=True)

        enable_running_stats(model)

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
        "stage5_mild_recover_gap_feedback_dynamic_rho_lr_improved_sam"
        if adaptive
        else "stage5_mild_recover_gap_feedback_dynamic_rho_lr_sam"
    )

    method_label = (
        f"Stage5 Mild Recover Gap-aware Feedback Dynamic Rho+LR ImprovedSAM "
        f"(init_lr={args.lr_init}, init_rho={args.rho_init})"
        if adaptive
        else
        f"Stage5 Mild Recover Gap-aware Feedback Dynamic Rho+LR SAM "
        f"(init_lr={args.lr_init}, init_rho={args.rho_init})"
    )

    # Long tag is only used for printing and recording.
    exp_tag = (
        f"lrInit{args.lr_init}_lrRange{args.lr_min}-{args.lr_max}_"
        f"rhoInit{args.rho_init}_rhoRange{args.rho_min}-{args.rho_max}_"
        f"pat{args.feedback_patience}_gapPat{args.gap_patience}_"
        f"gapTh{args.gap_threshold}_"
        f"lrF{args.feedback_lr_factor}_rhoF{args.feedback_rho_factor}_"
        f"gapLrF{args.gap_lr_factor}_gapRhoF{args.gap_rho_factor}_"
        f"cool{args.cooldown}_"
        f"recPat{args.recover_patience}_"
        f"recLrF{args.recover_lr_factor}_recRhoF{args.recover_rho_factor}_"
        f"recGapTh{args.recover_gap_threshold}_"
        f"ls{args.label_smoothing}_"
        f"ap{args.adaptive_power}_"
        f"seed{args.seed}"
    )

    # Short tag is used for filenames to avoid "filename too long" errors.
    short_tag = (
        f"lr{args.lr_init}_rho{args.rho_init}_"
        f"gap{args.gap_threshold}_gpat{args.gap_patience}_"
        f"cool{args.cooldown}_rec{args.recover_patience}_"
        f"ls{args.label_smoothing}_s{args.seed}"
    )

    short_tag = make_safe_tag(short_tag)
    file_prefix = "stage5_mild_recover"

    print(f"\nRunning new Stage5 experiment: {method_label}")
    print(f"Experiment tag: {exp_tag}")
    print(f"File tag: {short_tag}")

    model = build_model("resnet18", num_classes=10).to(device)

    optimizer = build_feedback_sam_optimizer(
        model=model,
        lr=args.lr_init,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        rho=args.rho_init,
        adaptive=adaptive,
        eta=args.eta,
        exclude_bn_bias=True,
        adaptive_power=args.adaptive_power
    )

    controller = MildRecoverGapAwareFeedbackController(
        lr_init=args.lr_init,
        rho_init=args.rho_init,
        lr_min=args.lr_min,
        lr_max=args.lr_max,
        rho_min=args.rho_min,
        rho_max=args.rho_max,
        patience=args.feedback_patience,
        min_delta=args.feedback_min_delta,
        lr_factor=args.feedback_lr_factor,
        rho_factor=args.feedback_rho_factor,
        gap_threshold=args.gap_threshold,
        gap_lr_factor=args.gap_lr_factor,
        gap_rho_factor=args.gap_rho_factor,
        gap_patience=args.gap_patience,
        cooldown=args.cooldown,
        recover_patience=args.recover_patience,
        recover_lr_factor=args.recover_lr_factor,
        recover_rho_factor=args.recover_rho_factor,
        recover_gap_threshold=args.recover_gap_threshold,
    )

    history = []
    best_test_acc = 0.0
    best_epoch = 0
    best_model_path = ""

    for epoch in range(1, args.epochs + 1):
        current_lr = controller.lr
        current_rho = controller.rho

        optimizer.set_lr(current_lr)
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

        feedback_info = controller.step(
            train_acc=train_acc,
            val_acc=test_acc,
            val_loss=test_loss
        )

        next_lr = feedback_info["lr"]
        next_rho = feedback_info["rho"]
        adjusted = feedback_info["adjusted"]
        recovered = feedback_info["recovered"]
        reason = feedback_info["reason"]
        gap = feedback_info["gap"]

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch = epoch

            best_model_path = (
                f"results/models/{file_prefix}_{short_tag}_"
                f"ep{args.epochs}_best.pth"
            )

            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "optimizer": method_name,
                "method_label": method_label,
                "exp_tag": exp_tag,
                "file_tag": short_tag,
                "lr": current_lr,
                "rho": current_rho,
                "next_lr": next_lr,
                "next_rho": next_rho,
                "lr_init": args.lr_init,
                "lr_min": args.lr_min,
                "lr_max": args.lr_max,
                "rho_init": args.rho_init,
                "rho_min": args.rho_min,
                "rho_max": args.rho_max,
                "adaptive": adaptive,
                "adaptive_power": args.adaptive_power,
                "eta": args.eta,
                "label_smoothing": args.label_smoothing,
                "best_test_acc": best_test_acc,
                "seed": args.seed,
                "epochs": args.epochs,
            }, best_model_path)

        adjust_info = f" | Adjusted: {reason}" if adjusted else ""

        print(
            f"Epoch {epoch}/{args.epochs} | {method_name} | "
            f"LR: {current_lr:.6f} -> {next_lr:.6f} | "
            f"Rho: {current_rho:.6f} -> {next_rho:.6f} | "
            f"Gap: {gap * 100:.2f}% | "
            f"Train Acc: {train_acc * 100:.2f}% | "
            f"Test Acc: {test_acc * 100:.2f}% | "
            f"Best: {best_test_acc * 100:.2f}% | "
            f"Bad Loss Epochs: {feedback_info['bad_epochs']} | "
            f"Bad Gap Epochs: {feedback_info['bad_gap_epochs']} | "
            f"Good Epochs: {feedback_info['good_epochs']} | "
            f"Adjust Count: {feedback_info['adjust_count']} | "
            f"Recover Count: {feedback_info['recover_count']} | "
            f"Cooldown: {feedback_info['cooldown_counter']} | "
            f"Train Batches: {train_batches} | Eval Batches: {eval_batches} | "
            f"Train Time: {train_time:.2f}s | Eval Time: {eval_time:.2f}s | "
            f"Total: {epoch_time:.2f}s"
            f"{adjust_info}"
        )

        record = {
            "epoch": epoch,
            "optimizer": method_name,
            "method_label": method_label,
            "exp_tag": exp_tag,
            "file_tag": short_tag,
            "lr": current_lr,
            "rho": current_rho,
            "next_lr": next_lr,
            "next_rho": next_rho,
            "adjusted": adjusted,
            "recovered": recovered,
            "adjust_reason": reason,
            "generalization_gap": gap,
            "bad_epochs": feedback_info["bad_epochs"],
            "bad_gap_epochs": feedback_info["bad_gap_epochs"],
            "good_epochs": feedback_info["good_epochs"],
            "adjust_count": feedback_info["adjust_count"],
            "recover_count": feedback_info["recover_count"],
            "cooldown_counter": feedback_info["cooldown_counter"],
            "feedback_loss": test_loss,
            "best_feedback_loss": feedback_info["best_loss"],
            "lr_init": args.lr_init,
            "lr_min": args.lr_min,
            "lr_max": args.lr_max,
            "rho_init": args.rho_init,
            "rho_min": args.rho_min,
            "rho_max": args.rho_max,
            "gap_threshold": args.gap_threshold,
            "gap_patience": args.gap_patience,
            "recover_gap_threshold": args.recover_gap_threshold,
            "adaptive": adaptive,
            "adaptive_power": args.adaptive_power,
            "eta": args.eta,
            "label_smoothing": args.label_smoothing,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "best_test_acc": best_test_acc,
            "epoch_time": epoch_time,
        }

        history.append(record)

    log_path = (
        f"results/logs/{file_prefix}_{short_tag}_"
        f"ep{args.epochs}.csv"
    )

    pd.DataFrame(history).to_csv(log_path, index=False)
    print(f"Saved Stage5 log: {log_path}")

    summary_path = (
        f"results/tables/{file_prefix}_{short_tag}_"
        f"ep{args.epochs}_summary.csv"
    )

    pd.DataFrame([{
        "optimizer": method_name,
        "label": method_label,
        "exp_tag": exp_tag,
        "file_tag": short_tag,
        "best_epoch": best_epoch,
        "best_test_acc": best_test_acc,
        "best_model_path": best_model_path,
        "log_path": log_path,
        "lr_init": args.lr_init,
        "lr_min": args.lr_min,
        "lr_max": args.lr_max,
        "rho_init": args.rho_init,
        "rho_min": args.rho_min,
        "rho_max": args.rho_max,
        "feedback_patience": args.feedback_patience,
        "feedback_min_delta": args.feedback_min_delta,
        "feedback_lr_factor": args.feedback_lr_factor,
        "feedback_rho_factor": args.feedback_rho_factor,
        "gap_threshold": args.gap_threshold,
        "gap_patience": args.gap_patience,
        "gap_lr_factor": args.gap_lr_factor,
        "gap_rho_factor": args.gap_rho_factor,
        "cooldown": args.cooldown,
        "recover_patience": args.recover_patience,
        "recover_lr_factor": args.recover_lr_factor,
        "recover_rho_factor": args.recover_rho_factor,
        "recover_gap_threshold": args.recover_gap_threshold,
        "adaptive": adaptive,
        "adaptive_power": args.adaptive_power,
        "eta": args.eta,
        "label_smoothing": args.label_smoothing,
        "seed": args.seed,
        "epochs": args.epochs
    }]).to_csv(summary_path, index=False)

    print(f"Saved Stage5 summary: {summary_path}")

    return method_name, method_label, history, log_path


# =========================
# Plotting
# =========================

def plot_metric_comparison(
    stage5_label,
    stage5_history,
    prev_sam_history,
    prev_dynamic_rho_history,
    prev_stage4_history,
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

        epochs = [h["epoch"] for h in stage5_history]
        values = [h[metric_key] for h in stage5_history]

        if to_percent:
            values = [v * 100 for v in values]

        plt.plot(
            epochs,
            values,
            marker="o",
            linewidth=2,
            linestyle="-",
            markersize=4,
            label=stage5_label
        )

        if (
            prev_sam_history is not None
            and len(prev_sam_history) > 0
            and metric_key in prev_sam_history[0]
        ):
            prev_epochs = [h["epoch"] for h in prev_sam_history]
            prev_values = [h[metric_key] for h in prev_sam_history]

            if to_percent:
                prev_values = [v * 100 for v in prev_values]

            plt.plot(
                prev_epochs,
                prev_values,
                marker="x",
                linewidth=2,
                linestyle="-",
                markersize=4,
                label=args.prev_sam_label
            )

        if (
            prev_dynamic_rho_history is not None
            and len(prev_dynamic_rho_history) > 0
            and metric_key in prev_dynamic_rho_history[0]
        ):
            prev_epochs = [h["epoch"] for h in prev_dynamic_rho_history]
            prev_values = [h[metric_key] for h in prev_dynamic_rho_history]

            if to_percent:
                prev_values = [v * 100 for v in prev_values]

            plt.plot(
                prev_epochs,
                prev_values,
                marker="s",
                linewidth=2,
                linestyle="-",
                markersize=4,
                label=args.prev_dynamic_rho_label
            )

        if (
            prev_stage4_history is not None
            and len(prev_stage4_history) > 0
            and metric_key in prev_stage4_history[0]
        ):
            prev_epochs = [h["epoch"] for h in prev_stage4_history]
            prev_values = [h[metric_key] for h in prev_stage4_history]

            if to_percent:
                prev_values = [v * 100 for v in prev_values]

            plt.plot(
                prev_epochs,
                prev_values,
                marker="^",
                linewidth=2,
                linestyle="-",
                markersize=4,
                label=args.prev_stage4_label
            )

        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.title(f"Stage5 Mild Recover Gap-aware Feedback Comparison ({ylabel})")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        save_path = (
            f"results/figures/stage5_mild_recover_gap_feedback_metric_{metric_key}.png"
        )

        plt.savefig(save_path, dpi=300)
        plt.close()

        print(f"Saved metric figure: {save_path}")


def plot_lr_curve(stage5_history):
    epochs = [h["epoch"] for h in stage5_history]
    lrs = [h["lr"] for h in stage5_history]

    adjusted_epochs = [h["epoch"] for h in stage5_history if h["adjusted"]]
    adjusted_lrs = [h["lr"] for h in stage5_history if h["adjusted"]]

    recovered_epochs = [h["epoch"] for h in stage5_history if h["recovered"]]
    recovered_lrs = [h["lr"] for h in stage5_history if h["recovered"]]

    plt.figure(figsize=(9, 6))

    plt.plot(
        epochs,
        lrs,
        marker="o",
        linewidth=2,
        linestyle="-",
        markersize=4,
        label="Learning Rate"
    )

    if len(adjusted_epochs) > 0:
        plt.scatter(
            adjusted_epochs,
            adjusted_lrs,
            marker="x",
            s=80,
            label="Adjustment Point"
        )

    if len(recovered_epochs) > 0:
        plt.scatter(
            recovered_epochs,
            recovered_lrs,
            marker="s",
            s=80,
            label="Recovery Point"
        )

    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title("Stage5 Learning Rate Curve with Recovery")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    save_path = "results/figures/stage5_mild_recover_lr_curve.png"
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved LR curve: {save_path}")


def plot_rho_curve(stage5_history):
    epochs = [h["epoch"] for h in stage5_history]
    rhos = [h["rho"] for h in stage5_history]

    adjusted_epochs = [h["epoch"] for h in stage5_history if h["adjusted"]]
    adjusted_rhos = [h["rho"] for h in stage5_history if h["adjusted"]]

    recovered_epochs = [h["epoch"] for h in stage5_history if h["recovered"]]
    recovered_rhos = [h["rho"] for h in stage5_history if h["recovered"]]

    plt.figure(figsize=(9, 6))

    plt.plot(
        epochs,
        rhos,
        marker="o",
        linewidth=2,
        linestyle="-",
        markersize=4,
        label="Rho"
    )

    if len(adjusted_epochs) > 0:
        plt.scatter(
            adjusted_epochs,
            adjusted_rhos,
            marker="x",
            s=80,
            label="Adjustment Point"
        )

    if len(recovered_epochs) > 0:
        plt.scatter(
            recovered_epochs,
            recovered_rhos,
            marker="s",
            s=80,
            label="Recovery Point"
        )

    plt.xlabel("Epoch")
    plt.ylabel("Rho")
    plt.title("Stage5 Rho Curve with Recovery")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    save_path = "results/figures/stage5_mild_recover_rho_curve.png"
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved Rho curve: {save_path}")


def plot_gap_curve(stage5_history):
    epochs = [h["epoch"] for h in stage5_history]
    gaps = [h["generalization_gap"] * 100 for h in stage5_history]

    gap_threshold = stage5_history[0]["gap_threshold"] * 100
    recover_gap_threshold = stage5_history[0]["recover_gap_threshold"] * 100

    thresholds = [gap_threshold for _ in stage5_history]
    recover_thresholds = [recover_gap_threshold for _ in stage5_history]

    adjusted_epochs = [h["epoch"] for h in stage5_history if h["adjusted"]]
    adjusted_gaps = [
        h["generalization_gap"] * 100
        for h in stage5_history
        if h["adjusted"]
    ]

    plt.figure(figsize=(9, 6))

    plt.plot(
        epochs,
        gaps,
        marker="o",
        linewidth=2,
        linestyle="-",
        markersize=4,
        label="Generalization Gap"
    )

    plt.plot(
        epochs,
        thresholds,
        linewidth=2,
        linestyle="-",
        label="Gap Threshold"
    )

    plt.plot(
        epochs,
        recover_thresholds,
        linewidth=2,
        linestyle="-",
        label="Recovery Gap Threshold"
    )

    if len(adjusted_epochs) > 0:
        plt.scatter(
            adjusted_epochs,
            adjusted_gaps,
            marker="x",
            s=80,
            label="Adjustment / Recovery Point"
        )

    plt.xlabel("Epoch")
    plt.ylabel("Train Acc - Test Acc (%)")
    plt.title("Stage5 Generalization Gap Feedback Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    save_path = "results/figures/stage5_mild_recover_generalization_gap_curve.png"
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved gap curve: {save_path}")


def plot_feedback_loss_curve(stage5_history):
    epochs = [h["epoch"] for h in stage5_history]
    feedback_losses = [h["feedback_loss"] for h in stage5_history]

    adjusted_epochs = [h["epoch"] for h in stage5_history if h["adjusted"]]
    adjusted_losses = [h["feedback_loss"] for h in stage5_history if h["adjusted"]]

    plt.figure(figsize=(9, 6))

    plt.plot(
        epochs,
        feedback_losses,
        marker="o",
        linewidth=2,
        linestyle="-",
        markersize=4,
        label="Feedback Loss"
    )

    if len(adjusted_epochs) > 0:
        plt.scatter(
            adjusted_epochs,
            adjusted_losses,
            marker="x",
            s=80,
            label="Adjustment / Recovery Point"
        )

    plt.xlabel("Epoch")
    plt.ylabel("Feedback Loss")
    plt.title("Stage5 Feedback Loss and Adjustment Points")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    save_path = "results/figures/stage5_mild_recover_feedback_loss_curve.png"
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved feedback loss curve: {save_path}")


def save_all_summary(
    stage5_label,
    stage5_history,
    stage5_log_path,
    prev_sam_history,
    prev_dynamic_rho_history,
    prev_stage4_history,
    args
):
    rows = []

    if prev_sam_history is not None and len(prev_sam_history) > 0:
        best_sam_row = max(prev_sam_history, key=lambda h: h["test_acc"])
        rows.append({
            "method": args.prev_sam_label,
            "best_epoch": best_sam_row["epoch"],
            "best_test_acc": best_sam_row["test_acc"],
            "source": args.prev_sam_log
        })

    if prev_dynamic_rho_history is not None and len(prev_dynamic_rho_history) > 0:
        best_dynamic_rho_row = max(prev_dynamic_rho_history, key=lambda h: h["test_acc"])
        rows.append({
            "method": args.prev_dynamic_rho_label,
            "best_epoch": best_dynamic_rho_row["epoch"],
            "best_test_acc": best_dynamic_rho_row["test_acc"],
            "source": args.prev_dynamic_rho_log
        })

    if prev_stage4_history is not None and len(prev_stage4_history) > 0:
        best_stage4_row = max(prev_stage4_history, key=lambda h: h["test_acc"])
        rows.append({
            "method": args.prev_stage4_label,
            "best_epoch": best_stage4_row["epoch"],
            "best_test_acc": best_stage4_row["test_acc"],
            "source": args.prev_stage4_log
        })

    best_stage5_row = max(stage5_history, key=lambda h: h["test_acc"])
    rows.append({
        "method": stage5_label,
        "best_epoch": best_stage5_row["epoch"],
        "best_test_acc": best_stage5_row["test_acc"],
        "source": stage5_log_path
    })

    summary_path = "results/tables/stage5_mild_recover_gap_feedback_all_summary.csv"
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

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    prev_sam_history = load_history_csv(args.prev_sam_log, "previous SAM")

    prev_dynamic_rho_history = load_history_csv(
        args.prev_dynamic_rho_log,
        "previous Dynamic Rho ImprovedSAM"
    )

    prev_stage4_history = load_history_csv(
        args.prev_stage4_log,
        "previous Stage4 Dynamic Rho+LR ImprovedSAM"
    )

    method_name, stage5_label, stage5_history, stage5_log_path = run_experiment(
        args=args,
        train_loader=train_loader,
        test_loader=test_loader,
        criterion=criterion,
        device=device
    )

    plot_metric_comparison(
        stage5_label=stage5_label,
        stage5_history=stage5_history,
        prev_sam_history=prev_sam_history,
        prev_dynamic_rho_history=prev_dynamic_rho_history,
        prev_stage4_history=prev_stage4_history,
        args=args
    )

    plot_lr_curve(stage5_history)
    plot_rho_curve(stage5_history)
    plot_gap_curve(stage5_history)
    plot_feedback_loss_curve(stage5_history)

    save_all_summary(
        stage5_label=stage5_label,
        stage5_history=stage5_history,
        stage5_log_path=stage5_log_path,
        prev_sam_history=prev_sam_history,
        prev_dynamic_rho_history=prev_dynamic_rho_history,
        prev_stage4_history=prev_stage4_history,
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

    parser.add_argument("--lr_init", type=float, default=0.01)
    parser.add_argument("--rho_init", type=float, default=0.1)

    parser.add_argument("--lr_min", type=float, default=0.001)
    parser.add_argument("--lr_max", type=float, default=0.012)
    parser.add_argument("--rho_min", type=float, default=0.05)
    parser.add_argument("--rho_max", type=float, default=0.12)

    parser.add_argument("--feedback_patience", type=int, default=8)
    parser.add_argument("--feedback_min_delta", type=float, default=1e-4)
    parser.add_argument("--feedback_lr_factor", type=float, default=0.8)
    parser.add_argument("--feedback_rho_factor", type=float, default=0.9)

    parser.add_argument("--gap_threshold", type=float, default=0.05)
    parser.add_argument("--gap_patience", type=int, default=4)
    parser.add_argument("--gap_lr_factor", type=float, default=0.9)
    parser.add_argument("--gap_rho_factor", type=float, default=0.95)

    parser.add_argument("--cooldown", type=int, default=5)

    parser.add_argument("--recover_patience", type=int, default=5)
    parser.add_argument("--recover_lr_factor", type=float, default=1.03)
    parser.add_argument("--recover_rho_factor", type=float, default=1.02)
    parser.add_argument("--recover_gap_threshold", type=float, default=0.03)

    parser.add_argument("--eta", type=float, default=0.01)
    parser.add_argument("--adaptive_power", type=float, default=1.0)

    parser.add_argument(
        "--no_adaptive",
        action="store_true",
        help="Disable adaptive perturbation."
    )

    parser.add_argument(
        "--label_smoothing",
        type=float,
        default=0.05,
        help="Label smoothing value for CrossEntropyLoss."
    )

    parser.add_argument(
        "--prev_sam_log",
        type=str,
        default="results/logs/stage2_resnet18_cifar10_sam_lr0.01_ep100_seed42_rho0.1.csv"
    )

    parser.add_argument(
        "--prev_sam_label",
        type=str,
        default="Stage2 SAM Baseline"
    )

    parser.add_argument(
        "--prev_dynamic_rho_log",
        type=str,
        default=""
    )

    parser.add_argument(
        "--prev_dynamic_rho_label",
        type=str,
        default="Stage3 Dynamic Rho ImprovedSAM"
    )

    parser.add_argument(
        "--prev_stage4_log",
        type=str,
        default=""
    )

    parser.add_argument(
        "--prev_stage4_label",
        type=str,
        default="Stage4 Dynamic Rho+LR ImprovedSAM"
    )

    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug_batches", type=int, default=5)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_compare(args)