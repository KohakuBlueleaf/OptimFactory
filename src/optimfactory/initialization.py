import torch
import torch.nn as nn

from .utils import get_fan_in


def mup_init(params: list[torch.Tensor], is_output: bool = False) -> None:
    """µP-style initialization for a collection of parameters.

    Skips 1D tensors (biases, norm weights). For other tensors, uses
    `std = 1/sqrt(fan_in)`, where `fan_in = prod(shape[1:])`.

    Args:
        params: Iterable of parameter tensors to initialize.

    Returns:
        None.
    """
    for param in params:
        if param.ndim == 1:
            continue
        fan_in = get_fan_in(param)
        std = (1 / fan_in) ** (1 if is_output else 0.5)
        torch.nn.init.normal_(param, mean=0.0, std=std)


def mup_init_output(param: torch.Tensor) -> None:
    mup_init([param], is_output=True)


def mup_patch_output(output_layer: nn.Module, base_dim: int = 256) -> None:
    """Patch output layer to use µP-style initialization and logit fwd scaling.

    Args:
        output_layer: Output layer to patch.
        base_dim: Reference dimension used for scaling.

    Returns:
        None.
    """
    mup_init(output_layer.parameters(), is_output=True)
    for param in output_layer.parameters():
        if param.ndim > 1:
            fan_in = get_fan_in(param)
            break

    mup_out_scale = base_dim / fan_in

    def new_fwd(*args, **kwargs):
        return output_layer.org_forward(*args, **kwargs) * mup_out_scale

    output_layer.org_forward = output_layer.forward
    output_layer.forward = new_fwd


def mup_unpatch(output_layer: nn.Module) -> None:
    if not hasattr(output_layer, "org_forward"):
        return
    output_layer.forward = output_layer.org_forward
    del output_layer.org_forward
