"""Native LR schedulers.

This module used to depend on `diffusers.optimization` for
`TYPE_TO_SCHEDULER_FUNCTION`.  Since the low-dim FlowPolicy no longer needs
the `diffusers` package at all (and installing it pulls ~2 GB of extras),
we reimplement the small subset of schedulers we actually use, using only
`torch.optim.lr_scheduler.LambdaLR`.

Supported names (case-insensitive): `constant`, `constant_with_warmup`,
`linear`, `cosine`, `cosine_with_restarts`, `polynomial`.
"""

import math
from typing import Optional

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def _require(value, name, scheduler_name):
    if value is None:
        raise ValueError(
            f"{scheduler_name} scheduler requires `{name}`, please provide it.")
    return value


def _constant_lambda(_: int) -> float:
    return 1.0


def _constant_with_warmup_lambda(num_warmup_steps: int):
    def fn(step: int) -> float:
        if step < num_warmup_steps:
            return float(step) / float(max(1, num_warmup_steps))
        return 1.0
    return fn


def _linear_lambda(num_warmup_steps: int, num_training_steps: int):
    def fn(step: int) -> float:
        if step < num_warmup_steps:
            return float(step) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            float(num_training_steps - step)
            / float(max(1, num_training_steps - num_warmup_steps)),
        )
    return fn


def _cosine_lambda(num_warmup_steps: int, num_training_steps: int, num_cycles: float = 0.5):
    def fn(step: int) -> float:
        if step < num_warmup_steps:
            return float(step) / float(max(1, num_warmup_steps))
        progress = float(step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
    return fn


def _cosine_with_restarts_lambda(num_warmup_steps: int, num_training_steps: int, num_cycles: int = 1):
    def fn(step: int) -> float:
        if step < num_warmup_steps:
            return float(step) / float(max(1, num_warmup_steps))
        progress = float(step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps))
        if progress >= 1.0:
            return 0.0
        return max(0.0, 0.5 * (1.0 + math.cos(
            math.pi * ((float(num_cycles) * progress) % 1.0))))
    return fn


def _polynomial_lambda(num_warmup_steps: int, num_training_steps: int,
                       lr_end: float = 1e-7, power: float = 1.0, lr_init: float = 1.0):
    def fn(step: int) -> float:
        if step < num_warmup_steps:
            return float(step) / float(max(1, num_warmup_steps))
        if step > num_training_steps:
            return lr_end / lr_init
        lr_range = lr_init - lr_end
        decay_steps = num_training_steps - num_warmup_steps
        pct_remaining = 1.0 - (step - num_warmup_steps) / max(1, decay_steps)
        decay = lr_range * (pct_remaining ** power) + lr_end
        return decay / lr_init
    return fn


_SCHEDULER_FNS = {
    'constant': _constant_lambda,
    'constant_with_warmup': _constant_with_warmup_lambda,
    'linear': _linear_lambda,
    'cosine': _cosine_lambda,
    'cosine_with_restarts': _cosine_with_restarts_lambda,
    'polynomial': _polynomial_lambda,
}


def get_scheduler(name: str,
                  optimizer: Optimizer,
                  num_warmup_steps: Optional[int] = None,
                  num_training_steps: Optional[int] = None,
                  last_epoch: int = -1,
                  **kwargs):
    """Return a LambdaLR wrapping the requested schedule."""
    key = str(name).lower()
    if key not in _SCHEDULER_FNS:
        raise ValueError(
            f"Unknown scheduler '{name}'. Supported: {sorted(_SCHEDULER_FNS)}")

    if key == 'constant':
        lr_lambda = _constant_lambda
    elif key == 'constant_with_warmup':
        lr_lambda = _constant_with_warmup_lambda(
            _require(num_warmup_steps, 'num_warmup_steps', key))
    elif key in ('linear', 'cosine', 'cosine_with_restarts', 'polynomial'):
        nw = _require(num_warmup_steps, 'num_warmup_steps', key)
        nt = _require(num_training_steps, 'num_training_steps', key)
        if key == 'linear':
            lr_lambda = _linear_lambda(nw, nt)
        elif key == 'cosine':
            lr_lambda = _cosine_lambda(nw, nt, **kwargs)
        elif key == 'cosine_with_restarts':
            lr_lambda = _cosine_with_restarts_lambda(nw, nt, **kwargs)
        else:
            lr_lambda = _polynomial_lambda(nw, nt, **kwargs)
    else:
        raise AssertionError(key)

    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)
