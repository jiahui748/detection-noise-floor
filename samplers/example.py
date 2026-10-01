#!/usr/bin/env python3
"""Compare the native sampling operator with the deterministic one.

Run:  python example.py
Both operators are exercised in the same way: a forward pass to record the
disagreement between them, then repeated identical backward passes to test whether
each is reproducible against itself.
"""
import torch
import torch.nn.functional as F

from det_deform_attn import det_grid_sample_bilinear

torch.manual_seed(0)
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'

# a small tensor shaped like one level of a deformable attention head
B, C, H, W = 2, 32, 64, 64
value = torch.randn(B, C, H, W, device=DEV, dtype=torch.float32)
grid = torch.rand(B, H, W, 2, device=DEV, dtype=torch.float32) * 2 - 1

print('device:', DEV)
print('value :', tuple(value.shape), ' grid:', tuple(grid.shape))


def run_backward(fn, reps=8):
    """Repeat the identical backward and return the per-element spread across reps."""
    grads = []
    for _ in range(reps):
        v = value.clone().requires_grad_(True)
        fn(v, grid).sum().backward()
        grads.append(v.grad.detach().clone())
    return torch.stack(grads), grads[0]


print()
print('--- 1. do the two operators agree? ---')
v1 = value.clone().requires_grad_(True)
f_native = F.grid_sample(v1, grid, mode='bilinear',
                         padding_mode='zeros', align_corners=False)
f_native.sum().backward()
v2 = value.clone().requires_grad_(True)
f_ours = det_grid_sample_bilinear(v2, grid)
f_ours.sum().backward()
print('  forward max |diff|  : %.3e' % (f_native - f_ours).abs().max().item())
print('  gradient max |diff| : %.3e' % (v1.grad - v2.grad).abs().max().item())

print()
print('--- 2. is each operator reproducible against itself? ---')
for name, fn in (('F.grid_sample', F.grid_sample), ('deterministic gather', det_grid_sample_bilinear)):
    stack, first = run_backward(fn, reps=8)
    spread = (stack - first).abs().max().item()
    print('  %-22s repeated-backward max spread : %.3e  %s'
          % (name, spread, 'REPRODUCIBLE' if spread == 0.0 else 'NOT reproducible'))

print()
print('--- 3. strict mode ---')
torch.use_deterministic_algorithms(True)
for name, fn in (('F.grid_sample', F.grid_sample), ('deterministic gather', det_grid_sample_bilinear)):
    try:
        v = value.clone().requires_grad_(True)
        fn(v, grid).sum().backward()
        print('  %-22s : runs' % name)
    except RuntimeError as e:
        print('  %-22s : RuntimeError - %s' % (name, str(e)[:80]))
