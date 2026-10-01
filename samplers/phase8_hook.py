"""Phase 8: controlled Gaussian gradient noise after unscale+clip.

Design note (important for interpreting the dose):
    BaseTrainer.optimizer_step() = unscale_ -> clip_grad_norm_(0.1) -> step.
    Noise injected BEFORE clip_grad_norm_ can be rescaled away, so we replace
    optimizer_step with an equivalent implementation that injects AFTER the clip
    and immediately before scaler.step(). The dose is then exact.

The noise scale is relative to each tensor's RMS gradient, so it is
scale-free across layers. Target dose brackets the measured atomicAdd wobble
(~1e-9..1e-8 relative, from mech_gradnoise2.py).
"""
import torch


def install(sigma, verbose=True):
    from ultralytics.engine.trainer import BaseTrainer
    if getattr(BaseTrainer, "_phase8_installed", False):
        BaseTrainer._phase8_sigma = float(sigma)
        if verbose:
            print(f"[phase8] sigma updated to {sigma}")
        return

    def optimizer_step(self):
        self.scaler.unscale_(self.optimizer)                      # unscale
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
        s = float(getattr(self, "_phase8_sigma", 0.0))
        if s > 0:
            with torch.no_grad():
                for p in self.model.parameters():
                    if p.grad is None:
                        continue
                    rms = p.grad.pow(2).mean().sqrt().clamp_min(1e-12)
                    p.grad.add_(torch.randn_like(p.grad) * (s * rms))
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        if self.ema:
            self.ema.update(self.model)

    BaseTrainer.optimizer_step = optimizer_step
    BaseTrainer._phase8_installed = True
    BaseTrainer._phase8_sigma = float(sigma)
    if verbose:
        print(f"[phase8] gradient noise installed AFTER clip, sigma={sigma}")
