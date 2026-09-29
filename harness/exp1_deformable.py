#!/usr/bin/env python3
"""Experiment 1 -- architecture-level replication of the deformable-attention noise floor.

Second architecture: `DeformableDetrForObjectDetection` from the installed `transformers`
(5.17.0), on VisDrone2019-DET at 640 px.  The quantity of interest is the run-to-run
sigma across SAME-SEED repeats, not accuracy, so the schedule is short (10 epochs) by
design -- see audit/PREREGISTERED_EXPERIMENTS.md section 1.

ARMS
    a   F.grid_sample (the stock HF operator), strict mode OFF
    b   F.grid_sample, strict mode ON        (not part of the 9-run grid; kept for symmetry)
    c   deterministic gather sampler, strict mode OFF
    d   deterministic gather sampler, strict mode ON   <- expected bit-identical across repeats

The grid actually launched is (a) x6, (c) x3, (d) x3 = 9 runs, seed 0 everywhere.

VERIFICATION BUILT IN
    The script counts every `F.grid_sample` call during the first forward pass and records
    it in the result JSON as `grid_sample_calls_first_forward`, and records whether the
    monkey-patch was installed (`patch_installed`) and which module object it was applied
    to.  A run whose count is 0 in arm (a) or (d) is a broken run, not a null result.

USAGE
    python exp1_deformable.py --arm a --seed 0 --repeat 0 --epochs 10 \
        --data /root/autodl-tmp/mw/VisDrone --out /root/autodl-tmp/pr_exp/exp1/result
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path

# Hub kernels must be off: the `kernels` package is not installed and the box has no
# route to huggingface.co.  With USE_HUB_KERNELS on, the decorator on
# MultiScaleDeformableAttention would try to fetch a kernelised forward at call time.
os.environ.setdefault("USE_HUB_KERNELS", "0")
# Weights come from the reachable mirror, never from huggingface.co.
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
NUM_CLASSES = 10  # VisDrone2019-DET
CLASS_NAMES = ["pedestrian", "people", "bicycle", "car", "van", "truck",
               "tricycle", "awning-tricycle", "bus", "motor"]


# --------------------------------------------------------------------------------------
# determinism / seeding
# --------------------------------------------------------------------------------------
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def configure_determinism(strict: bool) -> dict:
    """Set the global backend flags for one arm and report what was actually set.

    Arm (a) must run with strict OFF: `torch.use_deterministic_algorithms(True)` raises
    for grid_sample's backward on CUDA.  Arms (c)/(d) replace that backward with a gather,
    which is what makes strict ON legal.  CUBLAS_WORKSPACE_CONFIG is set only for strict
    arms -- it is required for deterministic cuBLAS and it costs performance, so turning it
    on for arm (a) would misrepresent that arm.
    """
    info = {"strict_requested": bool(strict)}
    if strict:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)
        info["cublas_workspace_config"] = os.environ["CUBLAS_WORKSPACE_CONFIG"]
    else:
        torch.use_deterministic_algorithms(False)
        info["cublas_workspace_config"] = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    torch.backends.cudnn.deterministic = bool(strict)
    torch.backends.cudnn.benchmark = not strict
    # TF32 changes matmul precision; it must be OFF so all arms see the same arithmetic.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    info["use_deterministic_algorithms"] = torch.are_deterministic_algorithms_enabled()
    info["cudnn_deterministic"] = bool(torch.backends.cudnn.deterministic)
    info["cudnn_benchmark"] = bool(torch.backends.cudnn.benchmark)
    info["allow_tf32_matmul"] = bool(torch.backends.cuda.matmul.allow_tf32)
    info["allow_tf32_cudnn"] = bool(torch.backends.cudnn.allow_tf32)
    return info


# --------------------------------------------------------------------------------------
# the sampler switch
# --------------------------------------------------------------------------------------
def make_gather_grid_sample(det_impl, counter):
    """Adapt `det_grid_sample_bilinear(value, grid)` to F.grid_sample's signature.

    HF calls the operator as

        nn.functional.grid_sample(value_l_, sampling_grid_l_, mode="bilinear",
                                 padding_mode="zeros", align_corners=False)

    so the replacement has to accept and *validate* those keywords.  If HF ever changes
    the sampling mode, silently ignoring it would make the arm test a different operator
    than the one the paper identifies, so a mismatch raises instead.
    """
    def adaptive(value, grid, mode="bilinear", padding_mode="zeros",
                 align_corners=False, **kwargs):
        if mode != "bilinear" or padding_mode != "zeros" or align_corners is not False:
            raise ValueError(
                "deterministic gather sampler only implements bilinear / zeros / "
                f"align_corners=False, but was called with mode={mode!r}, "
                f"padding_mode={padding_mode!r}, align_corners={align_corners!r}")
        counter[0] += 1
        return det_impl(value, grid)
    return adaptive


class GridSamplePatcher:
    """Replace the grid_sample used INSIDE the HF deformable-attention module only.

    The HF call site is `modeling_deformable_detr.py:204`, i.e. the module does

        nn.functional.grid_sample(value_l_, sampling_grid_l_, mode="bilinear",
                                 padding_mode="zeros", align_corners=False)

    so it resolves `grid_sample` through the `torch.nn.functional` module attribute at
    call time.  Patching `torch.nn.functional.grid_sample` therefore reaches it.

    Two things could silently leave the patch inert, and both are handled explicitly:
      * a module that did `from torch.nn.functional import grid_sample` keeps its own
        reference, so the patch never reaches its call.  We scan the already-imported
        transformers modules related to deformable attention for an attribute bound to
        the original function and rebind those too.
      * a *different* call site (torchvision's resampler, for instance) would be rebound
        by a blanket sys.modules walk, which would make the arm change more than the
        deformable-attention operator.  The scan is therefore restricted to the
        Deformable-DETR / DINO modules; a match anywhere else is reported, not patched.
    """

    SCAN_PREFIXES = (
        "transformers.models.deformable_detr",
        "transformers.models.dino",
        "transformers.models.d_fine",
        "transformers.models.deimv2",
    )

    def __init__(self, deterministic_impl):
        self.original = F.grid_sample
        self._impl = deterministic_impl
        self.calls = 0
        self._counter = [0]
        self.replacement = None        # set on install
        self.adaptive = None           # the F.grid_sample-shaped replacement
        self.patched_modules: list[str] = []
        self.aliases_rebound: list[str] = []
        self.aliases_not_patched: list[str] = []   # other operators left untouched
        self._bound: list = []         # (module, attr, original_value)

    def _bind(self, mod, attr):
        old = getattr(mod, attr, None)
        if old is self.original:
            setattr(mod, attr, self.adaptive)
            self._bound.append((mod, attr, old))
            return True
        return False

    def _install(self, permanent: bool):
        self._counter[0] = 0
        self.replacement = make_gather_grid_sample(self._impl, self._counter)

        def adaptive(*args, **kwargs):
            before = self._counter[0]
            out = self.replacement(*args, **kwargs)
            self.calls += self._counter[0] - before
            return out

        self.adaptive = adaptive
        F.grid_sample = adaptive
        self._bound.append((F, "grid_sample", self.original))
        self.patched_modules.append("torch.nn.functional.grid_sample")

        for name, mod in list(sys.modules.items()):
            if mod is None or mod is F:
                continue
            if not any(name.startswith(p) for p in self.SCAN_PREFIXES):
                if getattr(mod, "grid_sample", None) is self.original:
                    # e.g. torchvision's resampler.  Record it so the run proves the
                    # arm did NOT silently change another operator, but do not patch it.
                    self.aliases_not_patched.append(f"{name}.grid_sample")
                continue
            if self._bind(mod, "grid_sample"):
                self.aliases_rebound.append(f"{name}.grid_sample")
        if permanent:
            self.patched_modules = [p + " (permanent)" for p in self.patched_modules]
        return self

    def __enter__(self):
        return self._install(permanent=False)

    def __exit__(self, *exc):
        for mod, attr, old in self._bound:
            try:
                setattr(mod, attr, old)
            except Exception:
                pass
        self._bound = []
        return False

    def install_permanent(self):
        """Arm (c)/(d): leave the gather sampler in place for the whole run."""
        return self._install(permanent=True)


# --------------------------------------------------------------------------------------
# hashing (bit-identity evidence)
# --------------------------------------------------------------------------------------
def hash_state_dict(state: dict) -> str:
    """sha256 over sorted parameter names + shapes + raw fp32 bytes.

    Identical hashing for every run is the whole point: two arms that produce the same
    digest on the same seed are bit-identical, and no tolerance argument is needed.
    """
    h = hashlib.sha256()
    for name in sorted(state.keys()):
        t = state[name]
        if not torch.is_tensor(t):
            continue
        t = t.detach().to("cpu")
        if t.is_floating_point():
            t = t.to(torch.float32).contiguous()
        else:
            t = t.contiguous()
        h.update(name.encode())
        h.update(str(tuple(t.shape)).encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()


# --------------------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------------------
class VisDroneYolo(Dataset):
    """YOLO layout -> the target dict HF detection losses want.

    YOLO labels are `class cx cy w h`, normalised.  HF's HungarianMatcher / loss consume
    `class_labels` (int64 [n]) and `boxes` (float [n,4], **normalised cxcywh**), the same
    convention, so the boxes are re-scaled and passed through -- never re-derived.
    """

    def __init__(self, root: Path, split: str, size: int = 640):
        self.img_dir = root / "images" / split
        self.lbl_dir = root / "labels" / split
        self.size = size
        self.files = sorted(p for p in self.img_dir.iterdir()
                            if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        if not self.files:
            raise RuntimeError(f"no images under {self.img_dir}")

    def __len__(self):
        return len(self.files)

    def _resize(self, im: Image.Image):
        return im.resize((self.size, self.size), Image.BILINEAR)

    def load_raw(self, i: int):
        """Return (original RGB array, targets in ORIGINAL pixel coords) -- used by eval
        so IoU is computed in a resolution-independent way and the size bins mean
        something in the source image's own units."""
        p = self.files[i]
        im = Image.open(p).convert("RGB")
        w0, h0 = im.size
        lbl = self.lbl_dir / (p.stem + ".txt")
        boxes, labels = [], []
        if lbl.exists():
            for line in lbl.read_text().splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                c = int(float(parts[0]))
                cx, cy, bw, bh = (float(v) for v in parts[1:5])
                if bw <= 0 or bh <= 0:
                    continue
                boxes.append([cx * w0, cy * h0, bw * w0, bh * h0])
                labels.append(c)
        return np.asarray(im), np.asarray(boxes, dtype=np.float32), np.asarray(labels, dtype=np.int64), (h0, w0)

    def __getitem__(self, i: int):
        p = self.files[i]
        im = Image.open(p).convert("RGB")
        w0, h0 = im.size
        im = self._resize(im)
        arr = np.asarray(im, dtype=np.float32) / 255.0
        arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
        pixel_values = torch.from_numpy(arr.transpose(2, 0, 1)).contiguous()

        lbl = self.lbl_dir / (p.stem + ".txt")
        boxes, labels = [], []
        if lbl.exists():
            for line in lbl.read_text().splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                c = int(float(parts[0]))
                cx, cy, bw, bh = (float(v) for v in parts[1:5])
                if bw <= 0 or bh <= 0:
                    continue
                # normalised cxcywh -- same numbers YOLO stores, simply carried across
                boxes.append([cx, cy, bw, bh])
                labels.append(c)

        target = {
            "class_labels": torch.tensor(labels, dtype=torch.int64),
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "orig_size": torch.tensor([h0, w0], dtype=torch.int64),
            "size": torch.tensor([self.size, self.size], dtype=torch.int64),
            "image_id": torch.tensor([i], dtype=torch.int64),
        }
        return pixel_values, target


def collate(batch):
    pixel_values = torch.stack([b[0] for b in batch], dim=0)
    labels = [b[1] for b in batch]
    return pixel_values, labels


# --------------------------------------------------------------------------------------
# evaluation (pycocotools, matching the paper's COCO-style protocol)
# --------------------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, ds: VisDroneYolo, device, batch_size: int, workers: int,
             num_workers_timeout: int = 120) -> dict:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    model.eval()
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers,
                        collate_fn=collate, pin_memory=True,
                        persistent_workers=False)

    results = []
    for bi, (pixel_values, labels) in enumerate(loader):
        pixel_values = pixel_values.to(device, non_blocking=True)
        outputs = model(pixel_values=pixel_values)
        logits = outputs.logits  # [B, Q, C+1]
        boxes = outputs.pred_boxes  # [B, Q, 4] normalised cxcywh
        probs = logits.sigmoid()
        for b in range(pixel_values.shape[0]):
            idx = int(labels[b]["image_id"].item())
            h0, w0 = labels[b]["orig_size"].tolist()
            pr = probs[b]                                   # [Q, C+1]; last = no-object
            cls_scores, cls_ids = pr[:, :-1].max(dim=-1)    # over the 10 real classes
            # restore source-image pixel coords: the paper's probe uses pixel-space
            # evaluation, and the size bins (32^2, 96^2) are defined in COCO pixels.
            bx = boxes[b]
            cx, cy, bw, bh = bx[:, 0] * w0, bx[:, 1] * h0, bx[:, 2] * w0, bx[:, 3] * h0
            xyxy = torch.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], dim=-1)
            xyxy = xyxy.clamp(min=0)
            xyxy[:, 2] = xyxy[:, 2].clamp(max=w0)
            xyxy[:, 3] = xyxy[:, 3].clamp(max=h0)
            wh = (xyxy[:, 2:] - xyxy[:, :2]).clamp(min=0)
            keep = (wh[:, 0] > 1.0) & (wh[:, 1] > 1.0)
            s = cls_scores[keep]
            if s.numel() == 0:
                continue
            top = min(100, s.numel())
            s, order = s.topk(top)
            c = cls_ids[keep][order]
            xy = xyxy[keep][order]
            for k in range(s.numel()):
                x0, y0, x1, y1 = xy[k].tolist()
                results.append({
                    "image_id": idx,
                    "category_id": int(c[k].item()) + 1,   # COCO ids are 1-based
                    "bbox": [x0, y0, x1 - x0, y1 - y0],
                    "score": float(s[k].item()),
                })
        if bi % 20 == 0:
            print(f"    [eval] batch {bi}/{len(loader)}", flush=True)

    if not results:
        return {"mAP50": 0.0, "mAP50_95": 0.0, "n_detections": 0}

    gt = []
    ann_id = 1
    for i in range(len(ds)):
        _, boxes_px, labels_i, (h0, w0) = ds.load_raw(i)
        for k in range(len(labels_i)):
            cx, cy, bw, bh = boxes_px[k].tolist()
            gt.append({
                "id": ann_id,
                "image_id": i,
                "category_id": int(labels_i[k]) + 1,
                "bbox": [cx - bw / 2, cy - bh / 2, bw, bh],
                "area": float(bw * bh),
                "iscrowd": 0,
            })
            ann_id += 1

    cats = [{"id": i + 1, "name": CLASS_NAMES[i]} for i in range(NUM_CLASSES)]
    gt_ds = {"images": [{"id": i} for i in range(len(ds))],
             "annotations": gt, "categories": cats}
    coco_gt = COCO()
    coco_gt.dataset = gt_ds
    coco_gt.createIndex()
    coco_dt = coco_gt.loadRes(results)

    out = {"n_detections": len(results), "n_gt": len(gt)}
    summaries = {}
    for name, area_rng, max_det in (("all", [0, 1e10], 100),
                                    ("small", [0, 32 ** 2], 100),
                                    ("medium", [32 ** 2, 96 ** 2], 100),
                                    ("large", [96 ** 2, 1e10], 100)):
        ev = COCOeval(coco_gt, coco_dt, "bbox")
        ev.params.areaRng = [area_rng]
        ev.params.areaRngLbl = [name]
        ev.params.maxDets = [1, 10, max_det]
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
        summaries[name] = {
            "mAP50_95": float(ev.stats[0]),
            "mAP50": float(ev.stats[1]),
            "mAP75": float(ev.stats[2]),
            "n_gt_area": int(ev.stats[3]) if not np.isnan(ev.stats[3]) else None,
        }
    out["mAP50"] = summaries["all"]["mAP50"]
    out["mAP50_95"] = summaries["all"]["mAP50_95"]
    out["mAP75"] = summaries["all"]["mAP75"]
    out["by_size"] = {k: {"mAP50": v["mAP50"], "mAP50_95": v["mAP50_95"]}
                      for k, v in summaries.items()}
    return out


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------
def build_model(args, device, log):
    from transformers import DeformableDetrConfig, DeformableDetrForObjectDetection

    if args.pretrained:
        log(f"[init] loading COCO weights via HF_ENDPOINT={os.environ['HF_ENDPOINT']}")
        model = DeformableDetrForObjectDetection.from_pretrained(
            args.pretrained, num_labels=NUM_CLASSES, ignore_mismatched_sizes=True)
        init_source = f"pretrained:{args.pretrained}"
    else:
        cfg = DeformableDetrConfig(num_labels=NUM_CLASSES)
        model = DeformableDetrForObjectDetection(cfg)
        init_source = "scratch"

    model.config.num_labels = NUM_CLASSES
    # unused by HF, but the module warns if absent
    model.config.disable_custom_kernels = True
    model.to(device)
    return model, init_source


def build_optimizer(model, base_lr, weight_decay, head_lr_mult):
    """AdamW with a separate, larger LR for the randomly initialised head.

    The COCO checkpoint's class head is 91-way and is reinitialised to VisDrone's 10
    classes, so it is the only part of the model that starts from noise.  Given the same
    LR as the pretrained backbone it contributes almost no movement while the backbone
    decays towards its initialisation, and the detector stays at untrained accuracy.
    Up-weighting the head is the standard fix when fine-tuning a detection model onto a
    new label set; the LR *ratio* is recorded in every result JSON.
    """
    head, rest = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (head if ("class_embed" in n or "bbox_embed" in n) else rest).append(p)
    groups = [
        {"params": rest, "lr": base_lr, "name": "backbone_transformer"},
        {"params": head, "lr": base_lr * head_lr_mult, "name": "box_head"},
    ]
    return torch.optim.AdamW(groups, lr=base_lr, weight_decay=weight_decay,
                             betas=(0.9, 0.999), eps=1e-8), len(head), len(rest)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["a", "b", "c", "d"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--repeat", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--head-lr-mult", type=float, default=10.0,
                    help="LR multiplier for the reinitialised class/box head")
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--max-grad-norm", type=float, default=0.0,
                    help="0 = no clipping.  Measured raw grad norm here is 58-175, so a "
                         "0.1 clip (the ultralytics default) scales the update down by "
                         "~1500x and the model does not train at all.")
    ap.add_argument("--warmup-steps", type=int, default=100)
    ap.add_argument("--size", type=int, default=640)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit-train", type=int, default=0, help="0 = all")
    ap.add_argument("--limit-val", type=int, default=0, help="0 = all")
    ap.add_argument("--data", default="/root/autodl-tmp/mw/VisDrone")
    ap.add_argument("--out", default="/root/autodl-tmp/pr_exp/exp1/result")
    ap.add_argument("--weights-dir", default="/root/autodl-tmp/pr_exp/exp1/weights")
    ap.add_argument("--pretrained", default="SenseTime/deformable-detr")
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--no-lr-scale", action="store_true",
                    help="use --lr as-is instead of scaling it with the epoch budget")
    ap.add_argument("--save-weights", action="store_true", default=True)
    ap.add_argument("--no-save-weights", dest="save_weights", action="store_false")
    ap.add_argument("--verify-steps", type=int, default=0,
                    help="if >0, stop after this many optimiser steps (smoke test)")
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile the deformable-attention modules.  Arm (d) spends "
                         "57 min/epoch because strict mode routes the gather sampler's "
                         "many small tensor ops through a slow deterministic path; "
                         "fusing them helps.  OFF by default because it must not change "
                         "the numerics, and that has to be verified before it is used.")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    strict = args.arm in ("b", "d")
    use_gather = args.arm in ("c", "d")
    if args.no_pretrained:
        args.pretrained = None

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    wdir = Path(args.weights_dir)
    name = args.tag or f"exp1_{args.arm}_s{args.seed}_r{args.repeat}"
    result_path = outdir / f"{name}.json"

    def log(msg):
        print(f"[{name}] {msg}", flush=True)

    log(f"START arm={args.arm} strict={strict} gather={use_gather} epochs={args.epochs} "
        f"batch={args.batch} size={args.size}")

    det_info = configure_determinism(strict)
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        log("WARNING: CUDA unavailable, running on CPU")

    # --- the sampler switch -----------------------------------------------------------
    sys.path.insert(0, "/root/autodl-tmp/mw")
    from det_deform_attn import det_grid_sample_bilinear  # noqa: E402

    patcher = GridSamplePatcher(det_grid_sample_bilinear)
    if use_gather:
        patcher.install_permanent()
        patch_installed = True
    else:
        patch_installed = False
    log(f"[patch] installed={patch_installed} modules={patcher.patched_modules} "
        f"aliases_rebound={patcher.aliases_rebound} "
        f"other_operators_left_alone={patcher.aliases_not_patched}")

    # numerical equivalence of the two samplers, recorded in the result JSON: the arm
    # is only a switch of implementation if the forward outputs agree to fp32 rounding.
    _rng = np.random.default_rng(12345)
    _v = torch.from_numpy(_rng.standard_normal((4, 32, 40, 40)).astype(np.float32)).to(device)
    _g = torch.from_numpy((_rng.random((4, 100, 4, 2)) * 2 - 1).astype(np.float32)).to(device)
    with torch.no_grad():
        # the stock reference comes from the operator we captured at import time, so it
        # is the true F.grid_sample even when the gather sampler is permanently installed
        _ref_out = patcher.original(_v, _g, mode="bilinear", padding_mode="zeros",
                                    align_corners=False)
        _alt_out = det_grid_sample_bilinear(_v, _g)
    sampler_forward_maxdiff = float((_ref_out - _alt_out).abs().max().item())
    log(f"[verify] gather vs F.grid_sample forward max|diff| = {sampler_forward_maxdiff:.3e}")

    model, init_source = build_model(args, device, log)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"[model] init={init_source} params={n_params/1e6:.2f}M")

    compile_info = {"enabled": bool(args.compile)}
    if args.compile:
        # Fusion only; the operator under test and its arithmetic are unchanged.  The
        # gather sampler's backward is what costs 57 min/epoch under strict mode, and it
        # is a long chain of small elementwise/mask ops that a fused kernel can collapse.
        try:
            model = torch.compile(model, mode="default")
            compile_info["backend"] = "inductor"
            log("[compile] torch.compile applied (fusion only; numerics must be re-verified "
                "against the uncompiled arm before this is trusted in a reported run)")
        except Exception as e:
            compile_info["error"] = f"{type(e).__name__}: {e}"
            log(f"[compile] FAILED, continuing uncompiled: {type(e).__name__}: {e}")

    # --- verify the patch is actually exercised ---------------------------------------
    # Arm (a): wrap a throw-away patcher around one forward and count.  Arm (c)/(d): the
    # patch is already permanent, so count the calls the existing wrapper forwards.
    probe = torch.zeros(1, 3, 64, 64, device=device)
    if use_gather:
        before = patcher.calls
        model.eval()
        with torch.no_grad():
            model(pixel_values=probe)
        calls_probe = patcher.calls - before
        model.train()
    else:
        with torch.no_grad():
            with GridSamplePatcher(det_grid_sample_bilinear) as gp:
                model.eval()
                model(pixel_values=probe)
                model.train()
        calls_probe = gp.calls
    log(f"[verify] grid_sample calls in one 64x64 forward = {calls_probe} "
        f"(prepatch hook reached: the patched name was the one invoked)")
    if calls_probe == 0:
        log("FATAL: the deformable-attention sampler was NOT exercised -- aborting rather "
            "than producing a meaningless run")
        result = {"status": "error", "error": "grid_sample_not_exercised",
                  "patched_modules": patcher.patched_modules, **vars(args)}
        result_path.write_text(json.dumps(result, indent=2, default=str))
        return 2

    # --- data -------------------------------------------------------------------------
    root = Path(args.data)
    train_ds = VisDroneYolo(root, "train", args.size)
    val_ds = VisDroneYolo(root, "val", args.size)
    if args.limit_train:
        train_ds.files = train_ds.files[: args.limit_train]
    if args.limit_val:
        val_ds.files = val_ds.files[: args.limit_val]
    log(f"[data] train={len(train_ds)} val={len(val_ds)}")

    def worker_init(wid):
        s = args.seed * 1000 + wid
        np.random.seed(s)
        random.seed(s)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=False, num_workers=args.workers,
        collate_fn=collate, pin_memory=True, drop_last=True,
        worker_init_fn=worker_init, persistent_workers=args.workers > 0)

    params = [p for p in model.parameters() if p.requires_grad]
    opt, n_head_tensors, n_rest_tensors = build_optimizer(
        model, args.lr, args.weight_decay, args.head_lr_mult)
    steps_per_epoch = max(1, len(train_loader) // max(1, args.accum))
    total_steps = steps_per_epoch * args.epochs
    warmup = max(1, min(args.warmup_steps, int(0.05 * total_steps)))

    # PEAK LR IS SCHEDULE-LENGTH AWARE.  A 1e-4 peak is the right peak for a 10-epoch
    # cosine, because it is reached at ~5% of the run and decayed continuously after
    # that.  Put the same peak on a 3-epoch run and the cosine traverses the whole decay
    # in 1200 steps: the LR is already 5e-9 by step 50, the loss stops moving (measured:
    # frozen at ~14.0 for two solid epochs), and a sigma measured on a model that is not
    # training is not evidence about the sampler.  Scale the peak with the budget so the
    # total LR mass delivered is comparable, and record both numbers in the JSON.
    ref_epochs = max(1, int(os.environ.get("EXP1_REF_EPOCHS", "10")))
    lr_scale = args.epochs / ref_epochs
    if lr_scale < 1.0 and not args.no_lr_scale:
        base_lr = args.lr * lr_scale
    else:
        base_lr = args.lr

    def lr_factor(step):
        """Multiplier applied on top of each group's own peak LR (LambdaLR semantics:
        the head group keeps its multiplier, it is not flattened)."""
        if step < warmup:
            return (step + 1) / warmup
        prog = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + np.cos(np.pi * min(1.0, prog)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
    log(f"[schedule] epochs={args.epochs} steps/epoch={steps_per_epoch} "
        f"total_steps={total_steps} warmup={warmup} peak_lr={base_lr:.3e} "
        f"head_lr={base_lr*args.head_lr_mult:.3e} (x{args.head_lr_mult}, "
        f"{n_head_tensors} head tensors) "
        f"(requested {args.lr:.3e}, scale {lr_scale:.3f} vs {ref_epochs}-epoch reference)")

    # --- train ------------------------------------------------------------------------
    # Snapshot the initial parameters so the run can prove it actually moved.  A sigma
    # measured on frozen weights is not evidence about the sampler, so the L2 norm of the
    # update is recorded and reported in the result JSON.
    init_snapshot = {k: v.detach().to("cpu").clone().float()
                     for k, v in model.state_dict().items() if v.is_floating_point()}
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model.train()
    epoch_times, losses = [], []
    step = 0
    t_start = time.time()
    stop = False
    for epoch in range(args.epochs):
        t0 = time.time()
        opt.zero_grad(set_to_none=True)
        run_loss, run_n = 0.0, 0
        for it, (pixel_values, labels) in enumerate(train_loader):
            pixel_values = pixel_values.to(device, non_blocking=True)
            labels = [{k: v.to(device) for k, v in t.items()} for t in labels]
            out = model(pixel_values=pixel_values, labels=labels)
            loss = out.loss
            (loss / args.accum).backward()
            run_loss += float(loss.detach())
            run_n += 1
            if (it + 1) % args.accum == 0:
                if args.max_grad_norm and args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % 50 == 0:
                    log(f"  ep{epoch} step{step}/{total_steps} "
                        f"loss={run_loss/max(1,run_n):.4f} lr={sched.get_last_lr()[0]:.2e}")
            if args.verify_steps and step >= args.verify_steps:
                stop = True
                break
        dt = time.time() - t0
        epoch_times.append(dt)
        losses.append(run_loss / max(1, run_n))
        log(f"epoch {epoch} done in {dt/60:.2f} min mean_loss={losses[-1]:.4f} "
            f"steps={step}")
        if stop:
            log(f"verify-steps reached, stopping after {step} steps")
            break

    train_wall = time.time() - t_start

    # --- final weights + hash ---------------------------------------------------------
    state = {k: v.detach().to("cpu") for k, v in model.state_dict().items()}
    digest = hash_state_dict(state)

    # how far did training actually move the weights?
    sq_delta = sq_norm = 0.0
    for k, v0 in init_snapshot.items():
        v1 = state[k].to("cpu").float()
        sq_delta += float(((v1 - v0) ** 2).sum())
        sq_norm += float((v0 ** 2).sum())
    weight_delta_l2 = math.sqrt(sq_delta)
    weight_init_l2 = math.sqrt(sq_norm)
    rel_weight_delta = weight_delta_l2 / weight_init_l2 if weight_init_l2 else None
    log(f"[movement] ||dtheta||={weight_delta_l2:.4f} ||theta0||={weight_init_l2:.2f} "
        f"relative={rel_weight_delta:.3e}")
    if rel_weight_delta is not None and rel_weight_delta < 1e-5:
        log("WARNING: training barely moved the weights -- a sigma from this run says "
            "little about the sampler.  Raise --lr or --epochs.")

    wpath = None
    if args.save_weights:
        wdir.mkdir(parents=True, exist_ok=True)
        wpath = wdir / f"{name}.pt"
        torch.save({"model": state, "args": vars(args), "sha256": digest}, wpath)
        log(f"[weights] {wpath} ({wpath.stat().st_size/1e6:.1f} MB) sha256={digest}")

    # --- eval -------------------------------------------------------------------------
    t0 = time.time()
    metrics = evaluate(model, val_ds, device, args.batch, args.workers)
    eval_wall = time.time() - t0
    log(f"[eval] mAP50={metrics.get('mAP50'):.5f} mAP50-95={metrics.get('mAP50_95'):.5f} "
        f"({eval_wall/60:.2f} min)")

    result = {
        "status": "ok",
        "name": name,
        "arm": args.arm,
        "arm_description": {
            "a": "F.grid_sample, strict off",
            "b": "F.grid_sample, strict on",
            "c": "deterministic gather sampler, strict off",
            "d": "deterministic gather sampler, strict on",
        }[args.arm],
        "sampler": "gather" if use_gather else "grid_sample",
        "strict": bool(strict),
        "seed": args.seed,
        "repeat": args.repeat,
        "epochs": args.epochs,
        "epochs_completed": len(epoch_times),
        "batch": args.batch,
        "accum": args.accum,
        "effective_batch": args.batch * args.accum,
        "lr": args.lr,
        "base_lr": base_lr,
        "head_lr": base_lr * args.head_lr_mult,
        "head_lr_mult": args.head_lr_mult,
        "lr_scale": lr_scale,
        "ref_epochs_for_lr": ref_epochs,
        "warmup_steps": warmup,
        "total_steps": total_steps,
        "max_grad_norm": args.max_grad_norm,
        "weight_delta_l2": weight_delta_l2,
        "weight_init_l2": weight_init_l2,
        "rel_weight_delta": rel_weight_delta,
        "weight_decay": args.weight_decay,
        "imgsz": args.size,
        "resize": "square_640_direct",
        "amp": False,
        "init": init_source,
        "n_params": n_params,
        "train_images": len(train_ds),
        "val_images": len(val_ds),
        "optimizer_steps": step,
        "epoch_seconds": epoch_times,
        "seconds_per_epoch": (sum(epoch_times) / len(epoch_times)) if epoch_times else None,
        "train_wall_seconds": train_wall,
        "eval_seconds": eval_wall,
        "mean_loss_per_epoch": losses,
        "final_loss": losses[-1] if losses else None,
        "gpu_peak_mem_MB": (torch.cuda.max_memory_allocated(device) / 1e6
                            if device.type == "cuda" else None),
        "mAP50": metrics.get("mAP50"),
        "mAP50_95": metrics.get("mAP50_95"),
        "mAP75": metrics.get("mAP75"),
        "by_size": metrics.get("by_size"),
        "n_detections": metrics.get("n_detections"),
        "n_gt": metrics.get("n_gt"),
        "weights_path": str(wpath) if wpath else None,
        "weights_sha256": digest,
        "hash_method": "sha256(sorted param names + shapes + raw fp32 bytes)",
        "grid_sample_calls_probe_forward": calls_probe,
        "grid_sample_calls_total": patcher.calls,
        "patch_installed": patch_installed,
        "patched_modules": patcher.patched_modules,
        "aliases_rebound": patcher.aliases_rebound,
        "other_grid_sample_operators_left_alone": patcher.aliases_not_patched,
        "sampler_forward_maxdiff_vs_grid_sample": sampler_forward_maxdiff,
        "determinism": det_info,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "transformers": __import__("transformers").__version__,
        "hf_endpoint": os.environ.get("HF_ENDPOINT"),
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "argv": sys.argv,
        "compile": compile_info,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "finished": True,
    }
    result_path.write_text(json.dumps(result, indent=2, default=str))
    log(f"[done] wrote {result_path}")
    if args.verify_steps:
        log("SMOKE TEST OK" if metrics.get("mAP50") is not None else "SMOKE TEST: no metrics")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # write a marker so the launcher does not silently retry
        import traceback
        traceback.print_exc()
        try:
            out = Path("/root/autodl-tmp/pr_exp/exp1/result")
            out.mkdir(parents=True, exist_ok=True)
            (out / "_last_error.json").write_text(json.dumps(
                {"status": "error", "error": f"{type(exc).__name__}: {exc}",
                 "argv": sys.argv, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")},
                indent=2))
        except Exception:
            pass
        sys.exit(1)
