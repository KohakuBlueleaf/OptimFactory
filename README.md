# OptimFactory

Small utilities to make PyTorch optimizer setup and µParam/µP‑style initialization easier.

This repo currently provides:

- **µP initialization helpers**
  - `mup_init(parameters)`: init all non‑bias tensors with std `1/sqrt(fan_in)`.
  - `mup_init_output(weight)`: output layer init with std `1/fan_in`.
- **µP parameter‑group factory**
  - `mup_param_group(parameters, base_lr, base_dim=256, weight_decay=1e-3, weight_decay_scale=True)`:
    builds param groups where learning rate and weight decay are scaled by fan‑in.
- **Optional param‑group splitting**
  - `muon_param_group_split(param_groups, dim_threshold=64)`:
    split groups for separate optimizers (e.g. Muon vs AdamW) based on tensor shape/fan‑in.

The code is intentionally lightweight and pure‑PyTorch.

## Install

From source:

```bash
pip install -e .
```

This package requires Python ≥3.10 and `torch` (and `torchvision` only if you run MNIST examples).

## Quick start

```python
import torch
import torch.nn as nn
import torch.optim as optim

from optimfactory import mup_init, mup_init_output, mup_param_group

model = nn.Sequential(
    nn.Linear(128, 512),
    nn.ReLU(),
    nn.Linear(512, 10),
)

# µP init: skip 1D bias tensors automatically
mup_init(model.parameters())
# output layer often uses a different scale
mup_init_output(model[-1].weight)

param_groups = mup_param_group(
    model.parameters(),
    base_lr=1e-3,
    base_dim=256,
    weight_decay=0.1,
    weight_decay_scale=True,
)

optimizer = optim.AdamW(param_groups, betas=(0.9, 0.98))
```

## API details

### `mup_init(params)`

Initializes each parameter tensor in `params`:

- if `param.ndim == 1` (bias / norm weight), leave untouched
- otherwise compute `fan_in = prod(param.shape[1:])`
- sample `N(0, 1/sqrt(fan_in))`

### `mup_init_output(param)`

Like `mup_init`, but uses std `1/fan_in`. Useful for final classifiers/heads.

### `mup_param_group(params, base_lr, base_dim=256, weight_decay=1e-3, weight_decay_scale=True)`

Builds param groups keyed by `(fan_in, ndim)` so same‑shaped tensors share hyper‑params.

Scaling rules:

- For 1D tensors: `lr_scale = 1`
- For others: `lr_scale = base_dim / fan_in`
- Group LR: `base_lr * lr_scale`
- Group WD:
  - if `weight_decay_scale=True`: `weight_decay / lr_scale`
  - else: fixed `weight_decay`

Returned value is a list of dicts suitable for any PyTorch optimizer.

### `muon_param_group_split(param_groups, dim_threshold=64)`

Given param groups (typically from `mup_param_group`), split into:

- `muon_group`: 2D tensors where `fan_in >= dim_threshold`
- `adam_group`: everything else

This is a convenience when you want to use a special optimizer for large matrices.
`optimfactory` does **not** ship an optimizer named “Muon”; if you use one, it’s from elsewhere.

### `ComboOptimizer(optimizers)` / `ComboLRScheduler(schedulers)`

Lightweight wrappers to treat multiple optimizers or LR schedulers as one object.

- `ComboOptimizer.step()` / `.zero_grad()` forward to each child optimizer.
- `ComboOptimizer` accepts optional `clip_grad_norm` and `grad_scaler` (`torch.amp.GradScaler`) for global clipping and AMP.
- `ComboLRScheduler.step()` forwards to each child scheduler.
- Both support `.state_dict()` and `.load_state_dict()` by storing child state dicts in a list.

## Examples

- `example/mnist.py`: MNIST CNN/MLP hybrid with µP init and µP‑scaled param groups.
  - It references `optim.Muon` and `anyschedule.AnySchedule`, which are **external**.
  - If you don’t have them installed, set `USE_MUON=False` or use the basic example below.
- `example/basic_usage.py`: minimal MLP training loop showing only optimfactory usage.

Running examples:

```bash
python example/basic_usage.py
python example/mnist.py
```

## Notes / roadmap

- The project is small; PRs for more init schemes, group rules, or example notebooks are welcome.
- If you want more µP theory background, search for “μParametrization / µP” papers and guides.
