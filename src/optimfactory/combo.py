"""Wrappers to operate on multiple optimizers/schedulers as one.

These are intentionally thin: they forward common calls to each wrapped
optimizer/scheduler, and store/load a combined state_dict.
"""

from __future__ import annotations

from typing import Any

import torch


class ComboOptimizer:
    """A lightweight wrapper that treats multiple optimizers as one.

    Args:
        optimizers: A list of PyTorch optimizers to be stepped/zeroed together.
        clip_grad_norm: If set, clip global grad norm over all parameters before stepping.
        grad_scaler: Optional `torch.amp.GradScaler` to integrate AMP scaling.
        clip_norm_type: Norm type used for clipping when `clip_grad_norm` is set.

    Notes:
        - `step()` and `zero_grad()` are forwarded to each optimizer in order.
        - If `grad_scaler` is provided, `step()` uses `scaler.step(opt)` and
          calls `scaler.update()` once after all optimizers.
        - `state_dict()` returns a dict with a list of child state_dicts.
        - `load_state_dict()` expects the same format.
        - `param_groups` is exposed as a concatenation of child param_groups
          for compatibility with some tooling, but schedulers should be wrapped
          with `ComboLRScheduler` instead of relying on this.
    """

    def __init__(
        self,
        optimizers: list[torch.optim.Optimizer],
        clip_grad_norm: float | None = None,
        grad_scaler: torch.amp.GradScaler | None = None,
        clip_norm_type: float = 2.0,
    ):
        if not optimizers:
            raise ValueError("ComboOptimizer requires a non-empty optimizer list.")
        self.optimizers = optimizers
        self.clip_grad_norm = clip_grad_norm
        self.clip_norm_type = clip_norm_type
        self.grad_scaler = grad_scaler

        # Cache unique parameters for global clipping.
        params: list[torch.Tensor] = []
        seen: set[int] = set()
        for opt in optimizers:
            for group in opt.param_groups:
                for p in group["params"]:
                    if p is None:
                        continue
                    pid = id(p)
                    if pid not in seen:
                        seen.add(pid)
                        params.append(p)
        self._params = params

    @property
    def param_groups(self) -> list[dict[str, Any]]:
        groups: list[dict[str, Any]] = []
        for opt in self.optimizers:
            groups.extend(opt.param_groups)
        return groups

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Zero gradients for all child optimizers.

        Args:
            set_to_none: Passed through to child `zero_grad`.

        Returns:
            None.
        """
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    def step(self, closure: Any | None = None) -> None:
        """Perform an optimization step for all child optimizers.

        Args:
            closure: Optional closure forwarded to each optimizer.

        Returns:
            None.
        """
        if self.clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self._params,
                max_norm=self.clip_grad_norm,
                norm_type=self.clip_norm_type,
            )

        scaler = self.grad_scaler
        if closure is None:
            if scaler is None:
                for opt in self.optimizers:
                    opt.step()
            else:
                for opt in self.optimizers:
                    scaler.step(opt)
                scaler.update()
        else:
            if scaler is None:
                for opt in self.optimizers:
                    opt.step(closure=closure)
            else:
                for opt in self.optimizers:
                    scaler.step(opt, closure=closure)
                scaler.update()

    def state_dict(self) -> dict[str, Any]:
        """Return the combined state dict.

        Returns:
            A dict containing a list of child optimizer state dicts.
        """
        return {
            "optimizers": [opt.state_dict() for opt in self.optimizers],
            "num_optimizers": len(self.optimizers),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load combined state dict into child optimizers.

        Args:
            state_dict: A dict produced by `state_dict()`.

        Returns:
            None.
        """
        child_states = state_dict.get("optimizers", [])
        if len(child_states) != len(self.optimizers):
            raise ValueError(
                f"Expected {len(self.optimizers)} optimizer states, got {len(child_states)}."
            )
        for opt, st in zip(self.optimizers, child_states, strict=True):
            opt.load_state_dict(st)


class ComboLRScheduler:
    """A wrapper that steps multiple learning-rate schedulers together.

    Args:
        schedulers: A list of PyTorch LR schedulers to step together.

    Notes:
        - `step()` is forwarded to each scheduler in order.
        - `state_dict()` returns a dict with a list of child state_dicts.
        - `load_state_dict()` expects the same format.
    """

    def __init__(self, schedulers: list[Any]):
        if not schedulers:
            raise ValueError("ComboLRScheduler requires a non-empty scheduler list.")
        self.schedulers = schedulers

    def step(self, *args: Any, **kwargs: Any) -> None:
        """Step all schedulers.

        Args:
            *args: Positional args forwarded to each scheduler.
            **kwargs: Keyword args forwarded to each scheduler.

        Returns:
            None.
        """
        for sch in self.schedulers:
            sch.step(*args, **kwargs)

    def state_dict(self) -> dict[str, Any]:
        """Return the combined state dict.

        Returns:
            A dict containing a list of child scheduler state dicts.
        """
        return {
            "schedulers": [sch.state_dict() for sch in self.schedulers],
            "num_schedulers": len(self.schedulers),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load combined state dict into child schedulers.

        Args:
            state_dict: A dict produced by `state_dict()`.

        Returns:
            None.
        """
        child_states = state_dict.get("schedulers", [])
        if len(child_states) != len(self.schedulers):
            raise ValueError(
                f"Expected {len(self.schedulers)} scheduler states, got {len(child_states)}."
            )
        for sch, st in zip(self.schedulers, child_states, strict=True):
            sch.load_state_dict(st)
