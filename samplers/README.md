# Deterministic sampling operators

`det_deform_attn.py` replaces the sampling operator used by multi-scale deformable
attention with one whose backward pass is deterministic, so that same-seed training
repeats become bit-identical.

## Why

The stock operator is `F.grid_sample`. Its forward pass is deterministic, but its
backward pass is not: it scatters gradients with `atomicAdd`, whose accumulation order
is not fixed. Two identical backward calls on identical inputs therefore differ, by
about 1.2e-9 to 9.9e-9 relative in our measurements.

The consequence for a training run is not small. Section 6.3 of the paper injects a
perturbation of that magnitude into the full detector and the trajectory leaves the
basin of its own control within three epochs.

## It is also a legality problem, not only a reproducibility one

Under `torch.use_deterministic_algorithms(True)`, `F.grid_sample` cannot run at all:

```
RuntimeError: grid_sampler_2d_backward_cuda does not have a deterministic implementation
```

We reproduced this on two independent architectures (RT-DETR through the ultralytics
fork, and Deformable-DETR through transformers), so it is a property of the operator
rather than of one codebase.

## Usage

```python
import torch
from det_deform_attn import det_grid_sample_bilinear, apply_deterministic_deform_attn

# 1. swap the operator for a single call
out = det_grid_sample_bilinear(value, grid)

# 2. or patch a whole model, including modules that imported grid_sample directly
apply_deterministic_deform_attn(model, verbose=True)
```

`example.py` runs both operators in strict mode and reports the forward and gradient
disagreement, then checks whether repeated backward calls agree with themselves.

## What it costs

End to end 1.03x wall clock when only the operator is replaced. If strict determinism
is also required, 1.8x: most of the cost is strict mode, not the sampler.

## The bound is for finite inputs only

The index bound that makes the integer gather safe is a proof for finite coordinates.
Clamping does not remove a NaN, and `NaN.long()` is `INT64_MIN`, which sends the gather
out of range. The shipped module sanitises the coordinates at the source rather than the
indices. If you copy this code, keep that ordering.
