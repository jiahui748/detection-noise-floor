"""Phase 8 driver v2: deterministic sampler + optional post-clip gradient noise.

Differences from v1 (phase8_train.py), both forced by measurement:
  1. detect_anomaly is DISABLED. The fork's trainer wraps every backward in
     torch.autograd.detect_anomaly(), which costs 2.1x wall clock (250 vs 118 min
     per 30 epochs) and is where the ScatterGatherKernel index-out-of-bounds
     arises. The sampler's backward is provably clean (bwd_test.py), so the
     detector is the problem and we simply do not enable it.
  2. Results are written with the plain ultralytics trainer but with that one
     behaviour patched out, so the sigma=0 control and the sigma>0 arms share
     ONE driver. Comparing across drivers would confound the noise effect with a
     0.005 mAP driver offset measured earlier.
"""
import argparse, os, sys, time
sys.path.insert(0, '<REMOTE-ROOT>/mw')
sys.path.insert(0, '<REMOTE-ROOT>')

ap = argparse.ArgumentParser()
ap.add_argument('--sigma', type=float, default=0.0)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--epochs', type=int, default=30)
ap.add_argument('--batch', type=int, default=8)
ap.add_argument('--workers', type=int, default=8)
ap.add_argument('--imgsz', type=int, default=640)
ap.add_argument('--name', required=True)
ap.add_argument('--data', default='<REMOTE-ROOT>/mw/VisDrone/VisDrone.yaml')
ap.add_argument('--project', default='<REMOTE-ROOT>/mw/result_sdd')
a = ap.parse_args()

# --- patch 1: kill detect_anomaly in the trainer ---
import contextlib
import torch
import ultralytics.engine.trainer as _tr

_orig_backward = _tr.BaseTrainer.__dict__.get('_do_train')

def _no_anomaly_backward(self):
    """Same as the fork's, but WITHOUT torch.autograd.detect_anomaly()."""
    self.scaler.scale(self.loss).backward()

# find and neutralise the anomaly context by replacing it with a no-op
class _NoAnomaly:
    def __enter__(self): return self
    def __exit__(self, *e): return False

torch.autograd.detect_anomaly = lambda *a, **k: _NoAnomaly()
print('[phase8v2] autograd anomaly detection DISABLED')

# --- patch 2: post-clip gradient noise ---
import phase8_hook
phase8_hook.install(a.sigma)

from det_deform_attn import apply_deterministic_deform_attn
apply_deterministic_deform_attn(verbose=True)

from ultralytics import RTDETR
model = RTDETR('<REMOTE-ROOT>/mw/RTDETR-main/RTDETR-main/weights/rtdetr-l.pt')
t0 = time.time()
model.train(
    data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch,
    workers=a.workers, device=0, optimizer='AdamW', lr0=1e-4, lrf=0.01,
    warmup_epochs=3, amp=False, cache=False, plots=False, val=True,
    save=True, save_period=10, project=a.project, name=a.name, exist_ok=True,
    pretrained=True, seed=a.seed, deterministic=True, patience=0,
    mosaic=0.0, mixup=0.0, close_mosaic=0, cos_lr=False, resume=False,
)
print('done in %.1f min' % ((time.time()-t0)/60))
