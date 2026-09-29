"""Experiment 2a: operator noise on REAL activations.

Closes the paper's limitation 9 ("operator noise was measured on synthetic tensors,
not on real activations"). Method: build the trained RT-DETR-L, spy on grid_sample,
push real VisDrone images through the model, capture the ACTUAL (value, grid) pairs
that the deformable attention feeds to the sampler, then repeat the identical
backward on those captured tensors and measure the spread.

Why a spy rather than hooks on named modules: the fork's module names are not stable
across versions, and grid_sample is exactly the operator under test, so intercepting
the call captures the right tensors by construction. Both import styles are patched,
because a module that did `from torch.nn.functional import grid_sample` keeps its own
reference.

Comparison: synthetic tensors, as reported in the paper (gradnoise.json), against real
activations at the same shapes and at the real shapes/densities the model uses.

Usage: python exp2a_realact.py --batch 8
"""
import argparse
import glob
import json
import sys
import types

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, '/root/autodl-tmp/mw')
sys.path.insert(0, '/root/autodl-tmp')

ap = argparse.ArgumentParser()
ap.add_argument('--batch', type=int, default=8)
ap.add_argument('--reps', type=int, default=8)
ap.add_argument('--limit', type=int, default=40)
ap.add_argument('--ckpt', default='/lwz/mw/result_sdd/pn_base_s0/weights/best.pt')
ap.add_argument('--out', default='/lwz/mw/pr_exp/realact.json')
a = ap.parse_args()

DEV = 'cuda'
torch.use_deterministic_algorithms(False)

captured = []
orig = F.grid_sample


def spy(value, grid, *ar, **kw):
    if len(captured) < a.limit and torch.is_tensor(value) and torch.is_tensor(grid):
        captured.append((value.detach().clone(), grid.detach().clone()))
    return orig(value, grid, *ar, **kw)


F.grid_sample = spy
patched = 0
for _n, mod in list(sys.modules.items()):
    if isinstance(mod, types.ModuleType) and getattr(mod, 'grid_sample', None) is orig:
        setattr(mod, 'grid_sample', spy)
        patched += 1
print(f'[2a] spied on grid_sample; also patched {patched} module-level references', flush=True)

from ultralytics import RTDETR

model = RTDETR(a.ckpt)
net = model.model.to(DEV).eval()
print('[2a] model loaded', flush=True)

files = sorted(glob.glob('/lwz/mw/VisDrone/images/val/*.jpg'))[: a.batch]
ims = []
for p in files:
    import cv2
    im = cv2.imread(p)
    im = cv2.resize(im, (640, 640))
    ims.append(torch.from_numpy(im[:, :, ::-1].copy()).permute(2, 0, 1).float() / 255.0)
x = torch.stack(ims).to(DEV)
print(f'[2a] real batch: {tuple(x.shape)} from {len(files)} VisDrone val images', flush=True)

with torch.no_grad():
    try:
        net(x)
    except Exception as e:                       # the model may expect a different arg form
        print('[2a] forward(x) failed:', type(e).__name__, e, flush=True)
        raise
print(f'[2a] captured {len(captured)} real (value, grid) pairs', flush=True)


def spread(fn, v, g, reps):
    outs = []
    for _ in range(reps):
        vv = v.detach().clone().requires_grad_(True)
        fn(vv, g).sum().backward()
        outs.append(vv.grad.detach().clone())
    return torch.stack(outs)


from det_deform_attn import det_grid_sample_bilinear

rows = []
seen = set()
for i, (v, g) in enumerate(captured):
    key = (tuple(v.shape), tuple(g.shape))
    if key in seen:                              # one per distinct shape
        continue
    seen.add(key)
    A = spread(orig, v, g, a.reps)
    B = spread(det_grid_sample_bilinear, v, g, a.reps)
    sa = float(A.std(dim=0).mean())
    sb = float(B.std(dim=0).mean())
    scale = float(A.mean().abs())
    rows.append(dict(idx=i, value_shape=list(v.shape), grid_shape=list(g.shape),
                     points=int(g.shape[1] * g.shape[2]),
                     grid_sd=sa, grid_rel=sa / max(scale, 1e-12),
                     ours_sd=sb, ours_rel=sb / max(scale, 1e-12),
                     grad_scale=scale, reps=a.reps))
    print(f'[2a] shape {tuple(v.shape)} grid {tuple(g.shape)} points={rows[-1]["points"]} '
          f'grid_sd={sa:.3e} rel={sa/max(scale,1e-12):.3e} ours_sd={sb:.3e}', flush=True)

synth = json.load(open('/root/autodl-tmp/figdata/gradnoise.json'))
json.dump({'real': rows, 'synthetic_reference': synth['A'], 'c_core_synthetic': synth['C'],
           'n_captured': len(captured), 'patched_modules': patched},
          open(a.out, 'w'), indent=1)
print(f'[2a] wrote {a.out}', flush=True)
print('[2a] synthetic reference rel-noise range: '
      f"{min(r['grid_rel'] for r in synth['A']):.3e} .. {max(r['grid_rel'] for r in synth['A']):.3e}",
      flush=True)
if rows:
    print('[2a] REAL-activation rel-noise range:      '
          f"{min(r['grid_rel'] for r in rows):.3e} .. {max(r['grid_rel'] for r in rows):.3e}",
          flush=True)
