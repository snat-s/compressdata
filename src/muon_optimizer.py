"""
Muon Optimizer - Momentum with Orthogonalization

The Muon optimizer is a variant of SGD with momentum that applies orthogonalization
to gradient updates. This helps maintain diversity in parameter updates and improves
training efficiency.

Developed for the NanoGPT speedrun by Keller Jordan et al.

Key features:
- Orthogonalized momentum updates
- Better than AdamW for transformer training
- Lower memory footprint
- Faster convergence

Reference: https://github.com/KellerJordan/modded-nanogpt
"""

import torch
from torch.optim import Optimizer
import torch.nn.functional as F


def zeropower_via_newtonschulz5(G, steps=5, eps=1e-7):
    """
    Newton-Schulz iteration for computing the zeroth power of a matrix.

    This is used to orthogonalize the gradient matrix.

    Args:
        G: input matrix
        steps: number of Newton-Schulz iterations
        eps: small constant for numerical stability

    Returns:
        Orthogonalized matrix
    """
    assert len(G.shape) == 2

    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()

    # Ensure matrix is positive definite
    X = X / (X.norm() + eps)

    # Perform the iteration in bfloat16 for speed
    if G.size(0) > G.size(1):
        X = X.T

    # Newton-Schulz iteration
    for _ in range(steps):
        A = X @ X.T
        B = A @ X
        X = a * X + b * B + c * A @ B

    if G.size(0) > G.size(1):
        X = X.T

    return X.to(G.dtype)


class Muon(Optimizer):
    """
    Muon optimizer: Momentum with orthogonalization

    This optimizer applies orthogonalization to momentum updates, which helps
    maintain diversity in parameter updates and improves convergence.

    Args:
        params: model parameters
        lr: learning rate (default: 0.02)
        momentum: momentum coefficient (default: 0.95)
        nesterov: whether to use Nesterov momentum (default: True)
        ns_steps: number of Newton-Schulz iterations for orthogonalization (default: 5)
        adamw_params: list of parameter names to use AdamW for instead of Muon

    Example:
        optimizer = Muon(model.parameters(), lr=0.02, momentum=0.95)
    """

    def __init__(
        self,
        params,
        lr=0.02,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        adamw_params=None,
        adamw_lr=3e-4,
        adamw_betas=(0.9, 0.95),
        adamw_eps=1e-8,
        adamw_wd=0.0,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0 or momentum > 1.0:
            raise ValueError(f"Invalid momentum value: {momentum}")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            adamw_lr=adamw_lr,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
            adamw_wd=adamw_wd,
        )

        self.adamw_params = set(adamw_params) if adamw_params else set()

        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step"""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            nesterov = group['nesterov']
            ns_steps = group['ns_steps']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                # Check if this parameter should use AdamW
                param_name = id(p)
                use_adamw = any(param_name == id(ap) for ap in self.adamw_params)

                if use_adamw:
                    # Use AdamW for this parameter
                    if len(state) == 0:
                        state['step'] = 0
                        state['exp_avg'] = torch.zeros_like(p)
                        state['exp_avg_sq'] = torch.zeros_like(p)

                    exp_avg = state['exp_avg']
                    exp_avg_sq = state['exp_avg_sq']
                    beta1, beta2 = group['adamw_betas']

                    state['step'] += 1

                    # Decay the first and second moment running average coefficient
                    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                    # Bias correction
                    bias_correction1 = 1 - beta1 ** state['step']
                    bias_correction2 = 1 - beta2 ** state['step']
                    step_size = group['adamw_lr'] / bias_correction1

                    # Compute norm
                    denom = (exp_avg_sq.sqrt() / (bias_correction2 ** 0.5)).add_(group['adamw_eps'])

                    # Apply weight decay
                    p.mul_(1 - group['adamw_lr'] * group['adamw_wd'])

                    # Update parameters
                    p.addcdiv_(exp_avg, denom, value=-step_size)

                else:
                    # Use Muon for this parameter
                    # Initialize momentum buffer
                    if 'momentum_buffer' not in state:
                        state['momentum_buffer'] = torch.zeros_like(grad)

                    buf = state['momentum_buffer']

                    # Apply orthogonalization to 2D parameters (weight matrices)
                    if grad.ndim >= 2:
                        # Reshape to 2D if needed
                        original_shape = grad.shape
                        if grad.ndim > 2:
                            grad_2d = grad.view(grad.shape[0], -1)
                        else:
                            grad_2d = grad

                        # Orthogonalize
                        grad_orth = zeropower_via_newtonschulz5(grad_2d, steps=ns_steps)

                        # Reshape back
                        if grad.ndim > 2:
                            grad = grad_orth.view(original_shape)
                        else:
                            grad = grad_orth

                    # Momentum update
                    buf.mul_(momentum).add_(grad)

                    # Nesterov momentum
                    if nesterov:
                        update = grad + momentum * buf
                    else:
                        update = buf

                    # Update parameters
                    p.add_(update, alpha=-lr)

        return loss


class MuonAdamW:
    """
    Hybrid optimizer that uses Muon for most parameters and AdamW for specific ones.

    This is useful for models where some parameters (like embeddings or layer norms)
    benefit from AdamW while others benefit from Muon.

    Args:
        muon_params: parameters to optimize with Muon
        adamw_params: parameters to optimize with AdamW
        muon_lr: learning rate for Muon (default: 0.02)
        adamw_lr: learning rate for AdamW (default: 3e-4)
        momentum: momentum for Muon (default: 0.95)
        betas: betas for AdamW (default: (0.9, 0.95))
    """

    def __init__(
        self,
        muon_params,
        adamw_params,
        muon_lr=0.02,
        adamw_lr=3e-4,
        momentum=0.95,
        betas=(0.0, 0.9999),  # beta1=0 for online/non-stationary learning (NNCP-style)
        eps=1e-8,
        weight_decay=0.0,
    ):
        # Create Muon optimizer
        self.muon = Muon(
            muon_params,
            lr=muon_lr,
            momentum=momentum,
            nesterov=True,
            ns_steps=5,
        )

        # Create AdamW optimizer
        self.adamw = torch.optim.AdamW(
            adamw_params,
            lr=adamw_lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step"""
        self.muon.step(closure)
        self.adamw.step(closure)

    def zero_grad(self, set_to_none=True):
        """Zero out the gradients"""
        self.muon.zero_grad(set_to_none=set_to_none)
        self.adamw.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        """Get optimizer state"""
        return {
            'muon': self.muon.state_dict(),
            'adamw': self.adamw.state_dict(),
        }

    def load_state_dict(self, state_dict):
        """Load optimizer state"""
        self.muon.load_state_dict(state_dict['muon'])
        self.adamw.load_state_dict(state_dict['adamw'])

    def set_lr(self, adamw_lr, muon_lr):
        """Update learning rates for both sub-optimizers."""
        for pg in self.adamw.param_groups:
            pg['lr'] = adamw_lr
        for pg in self.muon.param_groups:
            pg['lr'] = muon_lr


def configure_optimizers(model, muon_lr=0.02, adamw_lr=3e-4, weight_decay=0.0,
                         momentum=0.95, device_type='cuda'):
    """
    Configure optimizers for a model using Muon for weights and AdamW for others.

    This function separates parameters into two groups:
    - 2D parameters (weight matrices) -> Muon
    - 1D parameters (biases, norms, embeddings) -> AdamW

    Args:
        model: PyTorch model
        muon_lr: learning rate for Muon
        adamw_lr: learning rate for AdamW
        weight_decay: weight decay coefficient
        momentum: momentum for Muon optimizer
        device_type: 'cuda' or 'cpu'

    Returns:
        optimizer: configured hybrid optimizer
    """
    # Separate parameters by dimensionality
    muon_params = []
    adamw_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Use Muon for 2D+ parameters (matrices), AdamW for 1D (biases, norms)
        if param.ndim >= 2:
            muon_params.append(param)
        else:
            adamw_params.append(param)

    # Create hybrid optimizer
    optimizer = MuonAdamW(
        muon_params=muon_params,
        adamw_params=adamw_params,
        muon_lr=muon_lr,
        adamw_lr=adamw_lr,
        momentum=momentum,
        betas=(0.0, 0.9999),  # beta1=0 for online/non-stationary learning (NNCP-style)
        eps=1e-8,
        weight_decay=weight_decay,
    )

    return optimizer


if __name__ == "__main__":
    # Test the optimizer
    import torch.nn as nn

    # Create a simple model
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 10),
    )

    # Configure optimizer
    optimizer = configure_optimizers(model, muon_lr=0.02, adamw_lr=3e-4)

    # Dummy forward pass
    x = torch.randn(32, 10)
    y = torch.randn(32, 10)

    # Training step
    loss = F.mse_loss(model(x), y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    print("Muon optimizer test passed!")
    print(f"Loss: {loss.item():.4f}")
