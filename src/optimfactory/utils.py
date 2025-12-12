import torch


def get_fan_in(param: torch.Tensor) -> int:
    """Compute fan-in for a parameter tensor.

    For tensors shaped like (out, in, ...), this returns prod(in, ...).
    1D tensors (biases, norm weights) return -1.

    Args:
        param: Parameter tensor.

    Returns:
        Fan-in as an integer, or -1 for 1D tensors.
    """
    if param.ndim == 1:
        return -1
    fan_in = 1
    for dim in param.shape[1:]:
        fan_in *= dim
    return int(fan_in)
