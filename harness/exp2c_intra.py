"""Experiment 2c: characterise the EARLY transient.

The dose grid (exp2b) showed that after epoch 1 the per-step factor is ~1.0000 for
every dose and every seed: the divergence has already saturated. So the
amplification is a transient that finishes inside the first epoch, and the three
epoch-end snapshots cannot resolve it. This run snapshots every 100 optimizer steps
during epoch 1 and answers: over how many steps does the divergence go from eps to
its saturated value, i.e. what is the actual per-step rate?

One epoch only: the later trajectory is already in divergence.csv.

Usage: python exp2c_intra.py --eps 1e-9 --seed 0 --name intra_e1e-9_s0
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, '<REMOTE-ROOT>/mw')
sys.path.insert(0, '<REMOTE-ROOT>')

ap = argparse.ArgumentParser()
ap.add_argument('--eps', type=float, default=0.0)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--epochs', type=int, default=1)
ap.add_argument('--every', type=int, default=100)
ap.add_argument('--batch', type=int, default=8)
ap.add_argument('--workers', type=int, default=8)
ap.add_argument('--imgsz', type=int, default=640)
ap.add_argument('--name', required=True)
ap.add_argument('--snapdir', default='<REMOTE-ROOT>/mw/pr_exp/snap_intra')
ap.add_argument('--data', default='<REMOTE-ROOT>/mw/VisDrone/VisDrone.yaml')
ap.add_argument('--project', default='<REMOTE-ROOT>/mw/pr_exp/result_intra')
ap.add_argument('--perturb-seed', type=int, default=20260927)
a = ap.parse_args()

import torch

SNAPDIR = os.path.join(a.snapdir, a.name)
os.makedirs(SNAPDIR, exist_ok=True)


class _NoAnomaly:
    def __enter__(self):
        return self

    def __exit__(self, *e):
        return False


torch.autograd.detect_anomaly = lambda *ar, **kw: _NoAnomaly()
print('[exp2c] anomaly detection DISABLED', flush=True)

from det_deform_attn import apply_deterministic_deform_attn
apply_deterministic_deform_attn(verbose=True)

from ultralytics import RTDETR

model = RTDETR('<REMOTE-ROOT>/mw/RTDETR-main/RTDETR-main/weights/rtdetr-l.pt')

EPS = float(a.eps)
STATE = {'injected': False, 'rel': 0.0, 'n': 0, 'saved': []}


def inject(trainer):
    if STATE['injected']:
        return
    STATE['injected'] = True
    if EPS == 0.0:
        print('[exp2c] eps=0 control', flush=True)
        return
    num = den = 0.0
    with torch.no_grad():
        for p in trainer.model.parameters():
            if not p.requires_grad or not p.dtype.is_floating_point:
                continue
            rms = p.pow(2).mean().sqrt().clamp_min(1e-12)
            gen = torch.Generator(device=p.device).manual_seed(int(a.perturb_seed))
            z = torch.randn(p.shape, generator=gen, device=p.device, dtype=p.dtype)
            delta = z * (EPS * rms)
            p.add_(delta)
            num += float(delta.pow(2).sum())
            den += float(p.pow(2).sum())
    STATE['rel'] = (num ** 0.5) / max(den ** 0.5, 1e-30)
    ema = getattr(trainer, 'ema', None)
    if ema is not None and getattr(ema, 'ema', None) is not None:
        ema.ema.load_state_dict(trainer.model.state_dict())
    print(f'[exp2c] injected eps={EPS:g} rel={STATE["rel"]:.6e}', flush=True)


def snap(trainer):
    """Snapshot every `every` optimizer steps, plus the final step."""
    STATE['n'] += 1
    n = STATE['n']
    total = int(getattr(trainer, 'nb', 0) or 0)
    if n % a.every != 0 and n != total:
        return
    path = os.path.join(SNAPDIR, f'step{n:05d}.pt')
    sd = {k: v.detach().to('cpu', copy=True) for k, v in trainer.model.state_dict().items()}
    torch.save(sd, path)
    STATE['saved'].append({'step': n, 'file': os.path.basename(path)})
    print(f'[exp2c] snapshot step {n}/{total}', flush=True)


# on_train_batch_end fires after every optimizer step
model.add_callback('on_train_start', inject)
model.add_callback('on_train_batch_end', snap)

t0 = time.time()
model.train(
    data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch,
    workers=a.workers, device=0, optimizer='AdamW', lr0=1e-4, lrf=0.01,
    warmup_epochs=0, amp=False, cache=False, plots=False, val=False,
    save=False, project=a.project, name=a.name, exist_ok=True,
    pretrained=True, seed=a.seed, deterministic=True, patience=0,
    mosaic=0.0, mixup=0.0, close_mosaic=0, cos_lr=False, resume=False,
)
dt = (time.time() - t0) / 60.0
json.dump({'eps': EPS, 'seed': a.seed, 'every': a.every, 'wall_min': dt,
           'injected_rel': STATE['rel'], 'total_steps': STATE['n'],
           'saved': STATE['saved']}, open(os.path.join(SNAPDIR, 'meta.json'), 'w'), indent=1)
print(f'[exp2c] DONE eps={EPS:g} name={a.name} wall={dt:.1f}min steps={STATE["n"]}', flush=True)
