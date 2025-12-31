from typing import Any

import torch

from .utils import get_fan_in


def mup_param_group(
    params: list[torch.Tensor],
    base_lr: float,
    base_dim: int = 256,
    weight_decay: float = 1e-3,
    weight_decay_scale: bool = True,
    input_module: torch.nn.Module | None = None,
) -> list[dict]:
    """Create µP-scaled parameter groups for optimizers.

    Parameters are bucketed by (fan_in, ndim). Learning rate is scaled as:

    - ndim == 1: lr_scale = 1
    - else: lr_scale = base_dim / fan_in

    Weight decay is optionally scaled inversely to lr_scale.

    Args:
        params: Iterable of parameter tensors.
        base_lr: Base learning rate for fan-in == base_dim parameters.
        base_dim: Reference dimension used for scaling.
        weight_decay: Base weight decay.
        weight_decay_scale: If True, scale weight decay by `1/lr_scale`.

    Returns:
        List of optimizer parameter groups.
    """

    fan_in_groups = {}
    input_groups = {}

    if input_module is not None:
        input_params = set(input_module.parameters())
    else:
        input_params = set()

    for param in params:
        ndim = param.ndim
        fan_in = get_fan_in(param)
        if param in input_params:
            if (fan_in, ndim) not in input_groups:
                input_groups[(fan_in, ndim)] = {
                    "params": [],
                    "lr": base_lr,
                    "weight_decay": weight_decay,
                    "fan_in": fan_in,
                    "ndim": ndim,
                }
            input_groups[(fan_in, ndim)]["params"].append(param)

        if ndim == 1:
            lr_scale = 1
        else:
            lr_scale = base_dim / fan_in
        if (fan_in, ndim) not in fan_in_groups:
            fan_in_groups[(fan_in, ndim)] = {
                "params": [],
                "lr": base_lr * lr_scale,
                "weight_decay": (
                    weight_decay / lr_scale if weight_decay_scale else weight_decay
                ),
                "fan_in": fan_in,
                "ndim": ndim,
            }
        fan_in_groups[(fan_in, ndim)]["params"].append(param)

    return list(fan_in_groups.values()) + list(input_groups.values())


def moun_param_split(
    params: list[torch.Tensor],
    dim_threshold: int = 64,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    muon_params = []
    adam_params = []
    for param in params:
        if param.ndim == 2 and get_fan_in(param) >= dim_threshold:
            muon_params.append(param)
        else:
            adam_params.append(param)
    return muon_params, adam_params


def muon_param_group_split(
    param_groups: list[dict[str, torch.Tensor | float]] | list[torch.Tensor],
    dim_threshold: int = 64,
) -> tuple[list[dict], list[dict]]:
    """Split param groups into (muon_group, adam_group).

    Heuristic:
    - Send 2D tensors with fan_in >= dim_threshold to muon_group.
    - Everything else to adam_group.

    Args:
        param_groups: Param groups, usually from `mup_param_group`.
        dim_threshold: Fan-in cutoff for putting a 2D group into muon_group.

    Returns:
        Tuple of (muon_group, adam_group).
    """
    param_groups = list(param_groups)
    if isinstance(param_groups[0], torch.Tensor):
        return moun_param_split(param_groups, dim_threshold)
    muon_group = []
    adam_group = []
    for group in param_groups:
        ndim = group.get("ndim", group["params"][0].ndim)
        fan_in = group.get("fan_in", get_fan_in(group["params"][0]))
        if ndim == 2 and fan_in >= dim_threshold:
            muon_group.append(group)
        else:
            adam_group.append(group)
    return muon_group, adam_group


def compdp_param_group(
    modules: torch.nn.Module | list[torch.nn.Module],
    base_lr: float = 1e-3,
    base_wd: float = 1e-3,
    base_eps: float = 1e-6,
    base_beta1: float = 0.9,
    base_beta2: float = 0.99,
    base_dim: int = 256,
    depth_scale: float = 1.0,
    depth_alpha: float = 1.0,
    bs_scale: float = 1.0,
    ds_scale: float = 1.0,
    input_module: torch.nn.Module | None = None,
    output_module: torch.nn.Module | None = None,
) -> list[dict]:
    """Create Completed(d)P-scaled parameter groups for optimizers."""
    assert bs_scale > 0 and ds_scale > 0

    relative_bs = bs_scale / ds_scale
    beta1 = 1 - (1 - base_beta1) * relative_bs
    beta2 = 1 - (1 - base_beta2) * relative_bs

    fan_in_groups = {}
    input_groups = {}
    output_groups = {}

    general_params = []
    input_params = []
    output_params = []

    if not isinstance(modules, list):
        modules = [modules]

    for module in modules:
        for child in module.modules():
            if len(list(child.modules())) > 1:
                continue
            if child is input_module:
                input_params.extend(child.parameters())
            elif child is output_module:
                output_params.extend(child.parameters())
            else:
                general_params.extend(child.parameters())

    for param in input_params:
        ndim = param.ndim
        fan_in = get_fan_in(param)
        lr_scale = relative_bs**0.5
        if ndim == 1:
            eps_scale = relative_bs**-0.5
        else:
            eps_scale = base_dim / fan_in * (relative_bs**-0.5)
        if (fan_in, ndim) not in input_groups:
            input_groups[(fan_in, ndim)] = {
                "params": [],
                "lr": base_lr * relative_bs**0.5,
                "weight_decay": base_wd * relative_bs**0.5,
                "eps": base_eps * eps_scale,
                "betas": (beta1, beta2),
                "fan_in": fan_in,
                "ndim": ndim,
            }
        input_groups[(fan_in, ndim)]["params"].append(param)

    for param in general_params:
        ndim = param.ndim
        fan_in = get_fan_in(param)
        if ndim == 1:
            lr_scale = depth_scale ** (depth_alpha - 1) * (relative_bs**0.5)
            wd_scale = relative_bs**0.5
            eps_scale = depth_scale ** (depth_alpha - 1) * (relative_bs**-0.5)
        else:
            lr_scale = (
                base_dim
                / fan_in
                * depth_scale ** (depth_alpha - 1)
                * (relative_bs**0.5)
            )
            wd_scale = fan_in / base_dim * relative_bs**0.5
            eps_scale = base_dim / fan_in * depth_scale**-1 * (relative_bs**-0.5)

        if (fan_in, ndim) not in fan_in_groups:
            fan_in_groups[(fan_in, ndim)] = {
                "params": [],
                "lr": base_lr * lr_scale,
                "weight_decay": base_wd * wd_scale,
                "eps": base_eps * eps_scale,
                "betas": (beta1, beta2),
                "fan_in": fan_in,
                "ndim": ndim,
            }
        fan_in_groups[(fan_in, ndim)]["params"].append(param)

    for param in output_params:
        ndim = param.ndim
        fan_in = get_fan_in(param)
        eps_scale = relative_bs**-0.5
        if ndim == 1:
            lr_scale = relative_bs**0.5
            wd_scale = relative_bs**0.5
        else:
            lr_scale = base_dim / fan_in * (relative_bs**0.5)
            wd_scale = fan_in / base_dim * relative_bs**0.5

        if (fan_in, ndim) not in output_groups:
            output_groups[(fan_in, ndim)] = {
                "params": [],
                "lr": base_lr * lr_scale,
                "weight_decay": base_wd * wd_scale,
                "eps": base_eps * eps_scale,
                "betas": (beta1, beta2),
                "fan_in": fan_in,
                "ndim": ndim,
            }
        output_groups[(fan_in, ndim)]["params"].append(param)

    return (
        list(fan_in_groups.values())
        + list(input_groups.values())
        + list(output_groups.values())
    )
