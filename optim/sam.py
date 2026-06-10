import torch
from torch.optim import Optimizer


class SAM(Optimizer):
    """
    Sharpness-Aware Minimization optimizer.

    SAM performs two steps:
    1. Move weights to the neighborhood point with higher loss.
    2. Compute gradient at that point and update the original weights.
    """

    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        if rho < 0.0:
            raise ValueError(f"Invalid rho value: {rho}")

        defaults = dict(rho=rho, **kwargs)
        super().__init__(params, defaults)

        self.rho = rho
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()

        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None:
                    continue

                e_w = p.grad * scale.to(p.device)
                p.add_(e_w)
                self.state[p]["e_w"] = e_w

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                p.sub_(self.state[p]["e_w"])

        self.base_optimizer.step()

        if zero_grad:
            self.zero_grad()

    def zero_grad(self):
        self.base_optimizer.zero_grad()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device

        norms = []

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    norms.append(p.grad.norm(p=2).to(shared_device))

        if len(norms) == 0:
            return torch.tensor(0.0, device=shared_device)

        return torch.norm(torch.stack(norms), p=2)