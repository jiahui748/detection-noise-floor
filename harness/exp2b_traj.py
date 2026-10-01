"""Experiment 2b (trajectory version): full-model perturbation divergence, tracked.

Same design as exp2b.py, with one addition that the dose response needs: the
parameter state is snapshotted at the end of every epoch, so divergence can be read
as a function of step count rather than only at the endpoint.

Why snapshots are necessary
  The module-level measurement (Table 11) reads a per-step factor of 1.007-1.029 off
  a 600-step run, i.e. growth of 10^1 to 10^7. Over a 2400-step training segment the
  same factor saturates completely, so an endpoint-only measurement would show every
  dose collapsing to "large" and could not estimate the per-step factor at all.
  Three snapshots give three intervals, hence three independent estimates.

Divergence metric
  d(t) = ||theta_eps(t) - theta_0(t)|| / ||theta_0(t)|| computed against the control
  run of the SAME seed, so the only difference between the two trajectories is the
  injected perturbation. The per-step factor over [t1, t2] is
  (d(t2)/d(t1))^(1/(t2-t1)).

RNG discipline (unchanged): the perturbation comes from its own torch.Generator, so
the global stream that drives augmentation is never reset and control and perturbed
runs see identical data.
"""
import argparse
import sys
import time

sys.path.insert(0, '<REMOTE-ROOT>/mw')
sys.path.insert(0, '<REMOTE-ROOT>')

ap = argparse.ArgumentParser()
ap.add_argument('--eps', type=float, default=0.0)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--epochs', type=int, default=3)
ap.add_argument('--batch', type=int, default=8)
ap.add_argument('--workers', type=int, default=8)
ap.add_argument('--imgsz', type=int, default=640)
ap.add_argument('--name', required=True)
ap.add_argument('--snapdir', default='<REMOTE-ROOT>/mw/pr_exp/snap')
ap.add_argument('--data', default='<REMOTE-ROOT>/mw/VisDrone/VisDrone.yaml')
ap.add_argument('--project', default='<REMOTE-ROOT>/mw/pr_exp/result')
ap.add_argument('--perturb-seed', type=int, default=20260927)
a = ap.parse_args()

import json
import os

import torch

SNAPDIR = os.path.join(a.snapdir, a.name)
os.makedirs(SNAPDIR, exist_ok=True)


class _NoAnomaly:
    def __enter__(self):
        return self

    def __exit__(self, *e):
        return False


torch.autograd.detect_anomaly = lambda *ar, **kw: _NoAnomaly()
print('[exp2b-traj] autograd anomaly detection DISABLED', flush=True)

from det_deform_attn import apply_deterministic_deform_attn
apply_deterministic_deform_attn(verbose=True)

from ultralytics import RTDETR

model = RTDETR('<REMOTE-ROOT>/mw/RTDETR-main/RTDETR-main/weights/rtdetr-l.pt')

EPS = float(a.eps)
PSEED = int(a.perturb_seed)
STATE = {'injected': False, 'rel': 0.0, 'step': 0, 'snaps': []}


def inject(trainer):
    if STATE['injected']:
        return
    STATE['injected'] = True
    if EPS == 0.0:
        print('[exp2b-traj] eps=0 control: no perturbation, RNG untouched', flush=True)
        return
    num = den = 0.0
    with torch.no_grad():
        for p in trainer.model.parameters():
            if not p.requires_grad or not p.dtype.is_floating_point:
                continue
            rms = p.pow(2).mean().sqrt().clamp_min(1e-12)
            gen = torch.Generator(device=p.device).manual_seed(PSEED)
            z = torch.randn(p.shape, generator=gen, device=p.device, dtype=p.dtype)
            delta = z * (EPS * rms)
            p.add_(delta)
            num += float(delta.pow(2).sum())
            den += float(p.pow(2).sum())
    STATE['rel'] = (num ** 0.5) / max(den ** 0.5, 1e-30)
    ema = getattr(trainer, 'ema', None)
    if ema is not None and getattr(ema, 'ema', None) is not None:
        ema.ema.load_state_dict(trainer.model.state_dict())
    print(f'[exp2b-traj] injected eps={EPS:g} measured rel={STATE["rel"]:.6e}', flush=True)


def snap(trainer):
    """Snapshot the parameter state at the end of each epoch."""
    ep = int(getattr(trainer, 'epoch', -1)) + 1
    STATE['step'] += 1
    path = os.path.join(SNAPDIR, f'ep{ep}.pt')
    sd = {k: v.detach().to('cpu', copy=True) for k, v in trainer.model.state_dict().items()}
    torch.save(sd, path)
    STATE['snaps'].append({'epoch': ep, 'file': os.path.basename(path)})
    print(f'[exp2b-traj] snapshot epoch {ep} -> {path}', flush=True)


model.add_callback('on_train_start', inject)
model.add_callback('on_fit_epoch_end', snap)

t0 = time.time()
model.train(
    data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch,
    workers=a.workers, device=0, optimizer='AdamW', lr0=1e-4, lrf=0.01,
    warmup_epochs=3, amp=False, cache=False, plots=False, val=True,
    save=False, project=a.project, name=a.name, exist_ok=True,
    pretrained=True, seed=a.seed, deterministic=True, patience=0,
    mosaic=0.0, mixup=0.0, close_mosaic=0, cos_lr=False, resume=False,
)
dt = (time.time() - t0) / 60.0
with open(os.path.join(SNAPDIR, 'meta.json'), 'w') as f:
    json.dump({'eps': EPS, 'seed': a.seed, 'epochs': a.epochs, 'wall_min': dt,
               'injected_rel': STATE['rel'], 'snaps': STATE['snaps'],
               'iters_per_epoch': int(getattr(model.trainer, 'nb', 0) or 0)}, f, indent=1)
print(f'[exp2b-traj] DONE eps={EPS:g} seed={a.seed} name={a.name} wall={dt:.1f}min', flush=True)
