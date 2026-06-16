import torch
from torch.optim import Optimizer


class ImprovedSAM3(Optimizer):
    """
    ImprovedSAM3 optimizer.

    Supports:
    1. Standard SAM
    2. Adaptive perturbation
    3. Dynamic rho
    4. Dynamic learning rate
    5. BatchNorm / bias exclusion
    6. Safe parameter restoration

    adaptive=False:
        e_w = grad * rho / ||grad||

    adaptive=True:
        s = |w| + eta
        norm uses grad * s
        perturbation uses grad * s^adaptive_power
    """

    def __init__(
        self,
        params,
        base_optimizer,
        rho=0.05,
        adaptive=True,
        eta=0.01,
        eps=1e-12,
        exclude_bn_bias=True,
        adaptive_power=1.0,
        **kwargs
    ):
        if rho < 0.0:
            raise ValueError(f"Invalid rho value: {rho}")

        if eta < 0.0:
            raise ValueError(f"Invalid eta value: {eta}")

        if adaptive_power <= 0.0:
            raise ValueError(f"Invalid adaptive_power value: {adaptive_power}")

        defaults = dict(
            rho=rho,
            adaptive=adaptive,
            eta=eta,
            eps=eps,
            exclude_bn_bias=exclude_bn_bias,
            adaptive_power=adaptive_power,
            **kwargs
        )

        super().__init__(params, defaults)

        self.rho = rho
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)

        # Share param_groups and state with base optimizer.
        self.param_groups = self.base_optimizer.param_groups
        self.state = self.base_optimizer.state

    @staticmethod
    def _is_excluded_param(p):
        # Bias and BatchNorm parameters are usually 1-dimensional.
        return p.ndim <= 1

    def _first_param_device(self):
        for group in self.param_groups:
            for p in group["params"]:
                return p.device
        return torch.device("cpu")

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()

        if not torch.isfinite(grad_norm) or grad_norm <= 0:
            if zero_grad:
                self.zero_grad(set_to_none=True)
            return

        for group in self.param_groups:
            rho = group["rho"]
            eps = group["eps"]
            adaptive = group["adaptive"]
            eta = group["eta"]
            exclude_bn_bias = group["exclude_bn_bias"]
            adaptive_power = group["adaptive_power"]

            scale = rho / (grad_norm + eps)

            for p in group["params"]:
                if p.grad is None:
                    continue

                if exclude_bn_bias and self._is_excluded_param(p):
                    continue

                grad = p.grad

                if grad.is_sparse:
                    raise RuntimeError("ImprovedSAM3 does not support sparse gradients.")

                if adaptive:
                    s = p.detach().abs().add(eta)
                    e_w = grad * s.pow(adaptive_power) * scale.to(p.device)
                else:
                    e_w = grad * scale.to(p.device)

                p.add_(e_w)
                self.state[p]["sam_e_w"] = e_w

        if zero_grad:
            self.zero_grad(set_to_none=True)

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        # Restore all perturbed parameters.
        for group in self.param_groups:
            for p in group["params"]:
                e_w = self.state[p].pop("sam_e_w", None)
                if e_w is not None:
                    p.sub_(e_w)

        self.base_optimizer.step()

        if zero_grad:
            self.zero_grad(set_to_none=True)

    def step(self, closure=None):
        if closure is None:
            raise RuntimeError(
                "ImprovedSAM3 requires two forward-backward passes. "
                "Use first_step/second_step manually, or pass a closure."
            )

        with torch.enable_grad():
            loss = closure()

        self.first_step(zero_grad=True)

        with torch.enable_grad():
            closure()

        self.second_step(zero_grad=True)

        return loss

    def zero_grad(self, set_to_none=True):
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    def _grad_norm(self):
        shared_device = self._first_param_device()
        norm_sq = torch.zeros([], device=shared_device)

        for group in self.param_groups:
            adaptive = group["adaptive"]
            eta = group["eta"]
            exclude_bn_bias = group["exclude_bn_bias"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                if exclude_bn_bias and self._is_excluded_param(p):
                    continue

                grad = p.grad

                if grad.is_sparse:
                    raise RuntimeError("ImprovedSAM3 does not support sparse gradients.")

                # Norm uses grad * s.
                # adaptive_power only affects perturbation e_w.
                if adaptive:
                    s = p.detach().abs().add(eta)
                    grad = grad * s

                norm_sq += grad.detach().pow(2).sum().to(shared_device)

        return torch.sqrt(norm_sq)

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)

        self.base_optimizer.param_groups = self.param_groups
        self.base_optimizer.state = self.state

    def set_rho(self, rho):
        if rho < 0.0:
            raise ValueError(f"Invalid rho value: {rho}")

        self.rho = rho

        for group in self.param_groups:
            group["rho"] = rho

    def get_rho(self):
        return self.param_groups[0]["rho"]

    def set_eta(self, eta):
        if eta < 0.0:
            raise ValueError(f"Invalid eta value: {eta}")

        for group in self.param_groups:
            group["eta"] = eta

    def set_lr(self, lr):
        if lr < 0.0:
            raise ValueError(f"Invalid lr value: {lr}")

        for group in self.base_optimizer.param_groups:
            group["lr"] = lr

    def get_lr(self):
        return self.base_optimizer.param_groups[0]["lr"]