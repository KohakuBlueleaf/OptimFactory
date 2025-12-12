import torch

from .utils import get_fan_in


def mup_init(params: list[torch.Tensor]) -> None:
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
        std = (1 / fan_in) ** 0.5
        torch.nn.init.normal_(param, mean=0.0, std=std)


def mup_init_output(param: torch.Tensor) -> None:
    """µP-style output/head initialization.

    Uses std = 1/fan_in for the given weight tensor.

    Args:
        param: Weight tensor of the output/head layer.

    Returns:
        None.
    """
    fan_in = get_fan_in(param)
    std = 1 / fan_in
    torch.nn.init.normal_(param, mean=0.0, std=std)
