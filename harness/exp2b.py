"""Experiment 2b: perturbation divergence in the FULL RT-DETR-L.

Claim under test (DATA_AUDIT / paper Section 6.2-6.3): the operator's ~1e-9 relative
per-step noise is amplified by the optimisation at 1.007-1.029 per step, so that a
1e-9 hardware-level perturbation reaches observable scale. The submitted evidence
came from a standalone module (limitations 3 and 9 call this "the weakest link"),
not from full training.

Design
  * strict determinism (gather sampler + --deterministic) so that the ONLY source of
    divergence between two runs of the same seed is the injected perturbation;
  * inject eps at train start as a per-tensor RMS-relative perturbation;
  * train a short segment, save the final checkpoint;
  * compare final weights against the eps=0 run of the SAME seed and report
    d_final = ||theta_eps - theta_0|| / ||theta_0|| and the implied per-step factor
    (d_final/eps)^(1/steps), exactly as Table 11 does for the module.

RNG DISCIPLINE (this is the part that is easy to get wrong)
  The perturbation is drawn from its own torch.Generator. Calling torch.manual_seed()
  inside the hook would reset the global stream and make the control and the perturbed
  run see DIFFERENT augmentations, which would confound the dose with the data order.
  With eps=0 the generator is never touched at all.

Usage:
  python exp2b.py --eps 1e-9 --seed 0 --epochs 3 --name exp2b_e9_s0
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
ap.add_argument('--data', default='<REMOTE-ROOT>/mw/VisDrone/VisDrone.yaml')
ap.add_argument('--project', default='<REMOTE-ROOT>/mw/pr_exp/result')
ap.add_argument('--perturb-seed', type=int, default=20260927)
a = ap.parse_args()

import torch

# --- patch: the fork wraps every backward in detect_anomaly; the sampler's backward
# is provably clean, so we disable it exactly as phase8_train_v2.py does.
class _NoAnomaly:
    def __enter__(self):
        return self

    def __exit__(self, *e):
        return False


torch.autograd.detect_anomaly = lambda *ar, **kw: _NoAnomaly()
print('[exp2b] autograd anomaly detection DISABLED')

# --- strict determinism: swap in the gather-based sampler
from det_deform_attn import apply_deterministic_deform_attn
apply_deterministic_deform_attn(verbose=True)

from ultralytics import RTDETR

model = RTDETR('<REMOTE-ROOT>/mw/RTDETR-main/RTDETR-main/weights/rtdetr-l.pt')

EPS = float(a.eps)
PSEED = int(a.perturb_seed)
INJECTED = {'done': False, 'rel': 0.0}


def inject(trainer):
    """Perturb every trainable parameter by eps * its own RMS, once."""
    if INJECTED['done']:
        return
    INJECTED['done'] = True
    if EPS == 0.0:
        print('[exp2b] eps=0 control: no perturbation, no RNG touched')
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
    INJECTED['rel'] = (num ** 0.5) / max(den ** 0.5, 1e-30)
    # keep the EMA consistent with the perturbed model at t=0
    ema = getattr(trainer, 'ema', None)
    if ema is not None and getattr(ema, 'ema', None) is not None:
        ema.ema.load_state_dict(trainer.model.state_dict())
    print(f'[exp2b] injected eps={EPS:g}, measured relative ||dtheta||/||theta|| = '
          f'{INJECTED["rel"]:.6e}')


model.add_callback('on_train_start', inject)

t0 = time.time()
model.train(
    data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch,
    workers=a.workers, device=0, optimizer='AdamW', lr0=1e-4, lrf=0.01,
    warmup_epochs=3, amp=False, cache=False, plots=False, val=True,
    save=True, save_period=1, project=a.project, name=a.name, exist_ok=True,
    pretrained=True, seed=a.seed, deterministic=True, patience=0,
    mosaic=0.0, mixup=0.0, close_mosaic=0, cos_lr=False, resume=False,
)
dt = (time.time() - t0) / 60.0
print(f'[exp2b] DONE eps={EPS:g} seed={a.seed} name={a.name} '
      f'wall={dt:.1f}min injected_rel={INJECTED["rel"]:.6e}')
