# Experiment 1 — architecture-level replication on a SECOND deformable-attention detector

Harness for the preregistered Experiment 1 (`audit/PREREGISTERED_EXPERIMENTS.md` §1), run on the
AutoDL GPU box. It measures the run-to-run σ of a second architecture at a **fixed seed**, and
whether the deterministic gather sampler collapses that σ to zero with bit-identical weights.

| file | local | remote |
|---|---|---|
| training script | `pr_exp/exp1_deformable.py` | `<REMOTE-ROOT>/pr_exp/exp1_deformable.py` |
| grid launcher | `pr_exp/run_exp1.sh` | `<REMOTE-ROOT>/pr_exp/run_exp1.sh` |
| aggregator | `pr_exp/exp1_analyse.py` | `<REMOTE-ROOT>/pr_exp/exp1_analyse.py` |
| sampler switch | `audit/remote_json/exp2/det_deform_attn.py` (already on the box) | `<REMOTE-ROOT>/mw/det_deform_attn.py` |

---

## 0. DECLARED SCOPE BOUNDARY — read this before quoting any number from this experiment

**exp1 replicates the mechanism at aggregate-metric level. The per-size profile is OUT OF SCOPE.**

* The claim this experiment tests is the §5.4 **mechanism**: that the sampling operator's backward
  accumulation order is what makes fixed-seed runs differ, and that a deterministic gather sampler
  under strict mode removes that difference. That claim is fully testable at aggregate **mAP50 /
  mAP50-95 plus the weight hashes**, which is exactly what this grid measures.
* The **per-size noise floor** — and the finding that it peaks where the detector only partially
  solves the scale — is a VisDrone/RT-DETR result. Replicating it on Deformable-DETR would need a
  schedule long enough to populate the COCO medium and large bins: many more GPU-hours, answering a
  question this experiment does not ask. It is deliberately **not** authorised here.
* At the 6-epoch budget, `pycocotools` returns **−1 for every area bin except `all`**, so per-size σ is
  **undefined, not zero**. The distinction is the whole point: collapsing "undefined" into "zero" is
  precisely the error this paper exists to correct, so the aggregator excludes −1 sentinels from σ
  rather than averaging them, and the per-size table prints empty instead of printing 0.00000.
* Consequently: **do not report a per-size σ from exp1 in either direction.** An absent per-size
  result is a scope boundary; it is not evidence that per-size noise vanishes on this architecture.

---

## 1. Verification 1 — does the installed `transformers` actually call `F.grid_sample`?

**CONFIRMED, by reading the file.**

* Module: `<REMOTE-ROOT>/envs/mw/lib/python3.12/site-packages/transformers/models/deformable_detr/modeling_deformable_detr.py`
* Class: `MultiScaleDeformableAttention` (defined at line 172), instantiated as
  `DeformableDetrMultiscaleDeformableAttention.self.attn` (line 550), called from
  `encoder_layer.self_attn(...)` (line 707, the encoder self-attention) and the decoder
  cross-attention.
* **Exact call site — line 204**, inside `MultiScaleDeformableAttention.forward`:

```python
sampling_value_l_ = nn.functional.grid_sample(
    value_l_,
    sampling_grid_l_,
    mode="bilinear",
    padding_mode="zeros",
    align_corners=False,
)
```

Two things this changes relative to the earlier note in the preregistration:

1. The version installed here is `transformers 5.17.0`, which has **no function named
   `multi_scale_deformable_attention`**. The operator lives in a `nn.Module` subclass instead, which is
   why a grep for the function name missed it. The operator itself is unchanged: the same
   `value_l_` / `sampling_grid_l_` construction and the same `F.grid_sample` call with the same
   three keyword arguments as the ultralytics fork.
2. The call is written `nn.functional.grid_sample`, i.e. it resolves the attribute on the
   `torch.nn.functional` module **at call time**. That is what makes the monkey-patch work; had the
   module done `from torch.nn.functional import grid_sample` at import time it would have kept its
   own reference.

**Runtime proof, not just reading.** The script counts calls during one forward pass and records the
count. Measured: **48 `grid_sample` calls in a single 64×64 forward** (12 attention modules × 4 feature
levels). A run in which the count is 0 aborts with `status: "error"` rather than producing a number.
The smoke test recorded `grid_sample_calls_total = 11568` over 200 training steps.

The gather sampler and `F.grid_sample` were also compared numerically on real tensor shapes
(recorded in every result JSON as `sampler_forward_maxdiff_vs_grid_sample`):

| tensors | forward max abs diff | backward max abs diff |
|---|---|---|
| `[4,32,40,40]` × grid `[4,100,4,2]` | 1.00e-05 | 2.44e-06 |
| `[8,32,160,160]` × grid `[8,300,4,2]` | 2.35e-05 | 7.11e-06 |
| `[8,32,80,80]` × grid `[8,3008,4,2]` | 2.02e-05 | 8.58e-06 |

Padding semantics were checked explicitly against `padding_mode="zeros"` at out-of-range
coordinates (identical, including the zero cases). The gather replacement raises `ValueError` if it is
ever called with a mode/padding/alignment other than `bilinear`/`zeros`/`False`, so it cannot silently
change which operator is under test.

## 2. The sampler switch: how it patches HF, and what it does *not* touch

`GridSamplePatcher` in `exp1_deformable.py`:

* replaces `torch.nn.functional.grid_sample` (the attribute the HF module resolves at call time);
* additionally scans already-imported modules whose `grid_sample` attribute is bound to the original
  function, and rebinds them — but the scan is restricted to the Deformable-DETR / DINO /
  D-FINE / DEIMv2 module namespaces, so the arm changes the deformable-attention operator and
  nothing else;
* reports any other module found holding the original (`other_grid_sample_operators_left_alone`).
  In practice that is `torchvision.transforms._functional_tensor.grid_sample`; it is **deliberately
  left untouched**, and the result JSON records that fact;
* restores every rebound attribute on exit (arm (a) uses it as a context manager for the call-count
  probe only, so the remainder of arm (a) runs the stock operator).

`apply_deterministic_deform_attn()` from `det_deform_attn.py` was **not** used: it patches the
ultralytics fork by name and would have been inert against HF. Only `det_grid_sample_bilinear` is
imported from it, wrapped to accept `F.grid_sample`'s signature.

## 3. Verification 2 — weights

**Pretrained weights WERE obtained, from the mirror.** `HF_ENDPOINT=https://hf-mirror.com`
resolves `SenseTime/deformable-detr` (R50, COCO) and downloads `model.safetensors`
(160 769 532 bytes; the earlier stated 160 MB was the true `Content-Length`). Direct
`huggingface.co` is still blocked. `hustvl/deformable-detr` does **not** exist on the mirror (404).

Loading is `from_pretrained(..., num_labels=10, ignore_mismatched_sizes=True)`. The load report is
explicit and worth keeping in the record:

* `class_embed.0.weight` / `.bias` — **MISMATCH**, reinitialised from COCO's 91 classes to VisDrone's 10;
* four `...downsample.1.num_batches_tracked` buffers — UNEXPECTED, harmless (the backbone uses
  `DeformableDetrFrozenBatchNorm2d`).

So the replication starts from **COCO-pretrained backbone and transformer with a freshly initialised
10-class head** — the same footing as the paper's RT-DETR-L. This is recorded per run as
`init: "pretrained:SenseTime/deformable-detr"` (or `"scratch"` under `--no-pretrained`).

## 4. Arms, and the per-arm batch size (a documented deviation)

Seed is held at **0 in every run**; the quantity of interest is run-to-run jitter at a fixed seed.

| arm | sampler | strict (`torch.use_deterministic_algorithms`) | repeats |
|---|---|---|---|
| a | `F.grid_sample` | off | 6 |
| c | deterministic gather | off | 3 |
| d | deterministic gather | **on** | 3 |

9 runs. Order is a×6, c×3, d×3 so the primary σ exists before the secondary arms finish.

**Measured, with the card otherwise free** (10 steps after warm-up, real data, 640 px):

| arm | batch | accum | ms/step | steps/epoch | **min/epoch** | peak VRAM |
|---|---|---|---|---|---|---|
| a `grid_sample` strict off | 4 | 4 | 182 | 1617 | **4.89** | 18.0 GiB |
| a `grid_sample` strict **on** | 4 | 4 | — | — | **raises** | — |
| c gather strict off | 1 | 4 | 183 | 6471 | **19.7** | 9.5 GiB |
| d gather strict on | 1 | 4 | 532 | 6471 | **57.4** | 9.5 GiB |
| c gather strict off | 2 | 4 | 255 | 3235 | 13.7 | 18.1 GiB |
| d gather strict on | 2 | 4 | 988 | 3235 | 53.3 | 18.1 GiB |
| gather, strict off | 4 | 4 | — | — | **OOM** | — |

Two findings that are results in their own right:

* **Strict mode + `F.grid_sample` genuinely cannot run.** `RuntimeError: grid_sampler_2d_backward_cuda
  does not have a deterministic implementation`. This reproduces on the second architecture the exact
  limitation the ultralytics fork documents, and it is why arms (c)/(d) need the gather sampler at
  all. (A first probe at 128 px did *not* raise — at that size cuDNN picks a different kernel path.
  Only the 640 px probe is representative; do not trust the small-size probe.)
* **Strict mode, not the gather sampler, is the dominant cost**: at batch 2 the gather path is
  255 ms/step with strict off and 988 ms/step with strict on — a 3.9× slowdown from determinism alone.
  The gather sampler itself is ~1.4× the cost of `F.grid_sample` at equal batch.

**Deviation, stated plainly:** the gather sampler needs 18 GiB of activation memory at batch 2 and
OOMs at batch 4, while `F.grid_sample` fits batch 4 in 18 GiB. Arms therefore use different *micro*
batch sizes (a: 4, c/d: 1) but the **same effective batch of 4** via `--accum 4`. Batch size is not the
manipulated variable (sampler and strictness are), and every arm performs the same number of images
per optimiser update.

## 5. THE TRAINING-CONFIGURATION DEFECT (found and fixed before any result was used)

The first grid launch produced **contaminated results that must not be cited**, and the evidence is
worth keeping because it is the exact failure mode this experiment is designed to detect.

The script had inherited the ultralytics `clip_grad_norm_(params, 0.1)` default. Measured on this
model, the **raw gradient norm is 58–175**, so a 0.1 clip scaled every update by 0.0006–0.0017 —
roughly 1500× too much. Combined with a cosine schedule that fit the entire decay into 3 epochs
(1212 steps), the learning rate was already **5.1e-09 by step 50** and the model was frozen:

| | contaminated 3-epoch run | fixed run (this grid) |
|---|---|---|
| mean loss per epoch | 13.9465 → 13.8718 (**−0.5%**) | 2.9155 → 1.7659 (**−39%**) |
| peak LR at step 50 | 5.10e-09 | 6.00e-05 (head 6.00e-04) |
| `rel_weight_delta` | 1.83e-06 | 3.68e-02 |
| mAP50 | 0.00160 (untrained) | 0.0456 |
| same-seed weight hashes | c642e40c… vs 2c74a08e… (differ) | 506a66c8… vs c583cce9… (differ) |
| σ(mAP50) across the 2 repeats | **4.7e-08** | **6.60e-04** |

The σ from the frozen grid is 4.7e-08: not a measurement of a noise floor, a measurement of weight
decay. Reporting it would have let the paper claim "σ = 0 on a second architecture" for entirely the
wrong reason — the outcome §1's decision rule says forces a claim cut-back. Those two runs are kept
at `<REMOTE-ROOT>/pr_exp/exp1/contaminated/` as an auditable record, not deleted.

Three fixes, all now defaults and all recorded per run:

1. `--max-grad-norm 0` (no clipping). The measured norm is logged in the diagnostic; 0.1 was never
   appropriate for this model's loss scale.
2. **Peak LR scales with the epoch budget** (`base_lr`, `lr_scale`, `ref_epochs_for_lr`). `--lr 1e-4`
   is the 10-epoch peak; a 6-epoch run gets 6e-05. Warmup is 5% of total steps.
3. **The reinitialised head gets its own LR** (`--head-lr-mult`, default 10 → `head_lr`). The COCO
   checkpoint's 91-way class head is reinitialised to 10 classes, so it is the only part starting
   from noise; at the backbone's LR it contributes almost no movement.
4. **`rel_weight_delta` guard**: every run records ‖θ_final − θ_init‖/‖θ_init‖ and logs a WARNING
   below 1e-05, so a σ measured on a frozen model is now self-evident in the JSON.

## 6. THE PRIMARY MEASUREMENT (production configuration, full data)

Two same-seed repeats of arm (a), full 6471-image set, 6 epochs, production settings:

| run | mAP50 | mAP50-95 | `rel_weight_delta` | weight sha256 (head) |
|---|---|---|---|---|
| `check6_r0` | 0.04556721 | 0.02388495 | 3.679e-02 | `506a66c826d00ee8` |
| `check6_r1` | 0.04650124 | 0.02413121 | 3.524e-02 | `c583cce9c8b3a317` |
| **σ** | **6.604e-04** | **1.741e-04** | — | **2 distinct hashes** |

**The hashes differ, so arm (a) is NOT deterministic — σ > 0 on the second architecture.** The
a-versus-d contrast therefore survives: there is jitter to remove. σ(mAP50) = 6.6e-04 on a mean of
0.0460 is 1.4% relative. The `rel_weight_delta` WARNING stayed silent in both runs (3.7e-02 ≫ 1e-05):
the model is genuinely training, not frozen, so this σ is a measurement of the sampler rather than of
a static model.

## 7. Measured epoch times (full 6471-image set, production settings)

| arm | batch | accum | optimizer steps/epoch | **measured** | source |
|---|---|---|---|---|---|
| a | 4 | 4 | 404 | **5.06 min** | archived full epoch, 303.8 s, solo |
| c | 1 | 4 | 1617 | **19.83 min** | real epoch, 800 images, linear extrapolation |
| d | 1 | 4 | 1617 | **57.24 min** | real epoch, 800 images, linear extrapolation |

Arms c and d were measured by running a **real epoch** at production settings on 800 of the 6471
images and scaling by steps (736 ms/step and 2124 ms/step respectively). The earlier 12-step probe
figures (19.7 / 57.4 min) turned out to be accurate to within 0.7%, but they were not the basis of the
budget. Note the steps/epoch differ across arms (404 vs 1617) because `accum` keeps the effective
batch at 4 while the micro-batch must differ: the gather sampler needs ~10.3 GiB at batch 1 and OOMs
at batch 2 alongside anything else, while `F.grid_sample` fits batch 4 in 19.4 GiB.

**The launched grid's budget and cost** (agreed with the parent agent, sequential, one run at a time so
that GPU contention cannot perturb bit-identity):

| arm | epochs | repeats | measured h/run | total |
|---|---|---|---|---|
| a | 6 | 6 | 0.53 | 3.16 h |
| c | 6 | 3 | 1.98 | 5.95 h |
| d | 3 | **2** | 2.86 | 5.73 h |
| | | | | **14.8 h** |

Arm (d) is **2** repeats, not 3: bit-identity is established by two identical hashes, so the third was
redundancy costing 2.9 h against the server's **hard 06:00 auto-shutdown** — a 17.6 h grid on a
05:35 ETA left only 25 minutes of margin, which is not acceptable on a job this long. Arm (d) is also
shortened to 3 epochs for the same class of reason: its claim is **bit-identity of the weight hash**, a
property of the code path and not of training progress. The a-versus-c contrast of §7.3 *does* need one
shared budget, so arms a and c both run 6 epochs. Every JSON records its own `epochs` and the launcher
logs its own repeat counts, so a mixed grid is auditable in the results rather than only here.

**Per-run wall times are measured, not assumed.** Four arm-(a) runs completed on the real grid at
31, 31, 31 and 32 min (grid.log), against the 5.06 min/epoch basis. The launcher is **sequential on
purpose**: running repeats concurrently would halve the wall time but would put environment-dependent
contention inside the one arm whose claim is bit-identity, and the arm-(d) runs must have the card
otherwise idle.

**Incremental harvesting (non-negotiable).** No result is allowed to exist only on the remote disk:

* the launcher copies every result JSON to a **staging directory** on the workspace disk
  (`<REMOTE-ROOT>/pr_exp/exp1/harvest/`) at the end of *every* run, plus `exp1_sigma.json`;
* those JSONs are then pulled to the **local workspace** at `<WORKSPACE>\pr_exp\results\exp1\`;
* **the weights stay remote.** Each is 163 MB and the JSON carries the sha256, which *is* the
  bit-identity evidence, so pulling them would cost ~1.5 GB for no additional information;
* the launcher's own attempt to pull straight to the Windows workspace cannot work on this box, and
  says so rather than pretending: there is **no `python` on the remote PATH**, and the env python at
  `<REMOTE-ROOT>/envs/mw/bin/python` has no `paramiko`. Staging is the layer that matters; the local
  copy is made from the Windows side. Every completed JSON is already mirrored locally, and
  `EXP1_NOTES.md` is re-pulled whenever it changes.

## 8. Results contract

One JSON per run in `<REMOTE-ROOT>/pr_exp/exp1/result/exp1_<arm>_s<seed>_r<repeat>.json`, plus the
final weights in `<REMOTE-ROOT>/pr_exp/exp1/weights/`. Recorded per run:

`arm`, `arm_description`, `sampler`, `strict`, `seed`, `repeat`, `epochs`, `epochs_completed`,
`optimizer_steps`, `batch`, `accum`, `effective_batch`, `lr`, `imgsz`, `resize`, `amp: false`,
`init`, `n_params`, `mAP50`, `mAP50_95`, `mAP75`, `by_size` (all/small/medium/large, COCO area bins),
`n_detections`, `n_gt`, `epoch_seconds`, `seconds_per_epoch`, `mean_loss_per_epoch`,
`gpu_peak_mem_MB`, `weights_sha256`, `hash_method`, `grid_sample_calls_probe_forward`,
`grid_sample_calls_total`, `patch_installed`, `patched_modules`, `aliases_rebound`,
`other_grid_sample_operators_left_alone`, `sampler_forward_maxdiff_vs_grid_sample`,
`determinism` (requested vs actually-set flags), library versions, `argv`, `timestamp`.

The weight hash is the identity evidence: `sha256` over **sorted parameter names + shapes + raw fp32
bytes**, computed the same way in every run, so two runs with the same digest are bit-identical and no
tolerance argument is needed. Order-independence and 1e-8 sensitivity of the hash were both tested.

Protocol details fixed for every run: `imgsz 640`, direct square resize (bilinear) with ImageNet
normalisation, **no augmentation**, `shuffle=False` so the batch order is identical across repeats,
AMP off, TF32 off, AdamW with a cosine schedule whose peak is scaled to the epoch budget, **no gradient
clipping** (`--max-grad-norm 0`), a 10× LR multiplier on the reinitialised head, `drop_last=True`,
`torch.manual_seed`/numpy/random seeded, cuDNN benchmark/deterministic set per arm, and
`CUBLAS_WORKSPACE_CONFIG=:4096:8` **only** for the strict arms. Ground truth comes straight from the
YOLO labels: `class cx cy w h` normalised, passed through unchanged (verified against the raw label
file), never re-derived.

## 9. Exact commands

Smoke test (already run and passed — arm (d), 1 epoch, 200 images, wrote its JSON):
```bash
bash <REMOTE-ROOT>/pr_exp/scripts/smoke_d.sh
```

The grid as launched (this is the exact command that is running, pid 259587 from 12:00:15):
```bash
RUN_EPOCHS=6 RUN_EPOCHS_D=3 RUN_BATCH_A=4 RUN_BATCH_C=1 RUN_BATCH_D=1 \
RUN_HEAD_LR_MULT=10 RUN_MAX_GRAD_NORM=0 \
setsid bash <REMOTE-ROOT>/pr_exp/run_exp1.sh \
    > <REMOTE-ROOT>/pr_exp/exp1/logs/grid.out 2>&1 < /dev/null &
```

Resume an interrupted grid — **the same command**. Every run whose result JSON exists is skipped, so
re-running the launcher is always safe:
```bash
RUN_EPOCHS=6 RUN_EPOCHS_D=3 \
setsid bash <REMOTE-ROOT>/pr_exp/run_exp1.sh \
    >> <REMOTE-ROOT>/pr_exp/exp1/logs/grid.out 2>&1 < /dev/null &
```

Aggregate σ and the bit-identity check at any time, including mid-grid:
```bash
<REMOTE-ROOT>/envs/mw/bin/python <REMOTE-ROOT>/pr_exp/exp1_analyse.py \
    --result <REMOTE-ROOT>/pr_exp/exp1/result
```

Watch progress:
```bash
tail -f <REMOTE-ROOT>/pr_exp/exp1/logs/grid.log
tail -f <REMOTE-ROOT>/pr_exp/exp1/logs/exp1_a_s0_r0.log
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv
```

The launcher waits for free VRAM (`RUN_MIN_FREE_MIB`, default 19000) before each run and aborts the
grid rather than filling the disk (`RUN_MIN_FREE_GB`, default 3), because the card is shared with other
jobs on this box.

## 10. Arm (a) — COMPLETE, and this is the primary result

Arm (a), 6 same-seed repeats, **all sequential on an otherwise idle card** (this matters: contention
was deliberately kept out of the causal chain):

| run | status | mAP50 | mAP50-95 | `rel_weight_delta` | s/epoch | weight sha256 |
|---|---|---|---|---|---|---|
| `exp1_a_s0_r0` | complete (12:31, 31 min) | 0.040129 | 0.020983 | 3.730e-02 | 304.4 | `5506a8a59c535375` |
| `exp1_a_s0_r1` | complete (13:02, 31 min) | 0.045407 | 0.023460 | 3.800e-02 | 305.6 | `e65f2daafce14a8a` |
| `exp1_a_s0_r2` | complete (13:33, 31 min) | 0.045287 | 0.024649 | 3.629e-02 | 306.4 | `fcbd3adbcac94ef9` |
| `exp1_a_s0_r3` | complete (14:05, 31 min) | 0.047614 | 0.025035 | 3.683e-02 | 307.9 | `3ae65492fdd3967d` |
| `exp1_a_s0_r4` | complete (14:51, 31 min) | 0.044429 | 0.023711 | 3.623e-02 | 311.0 | `92267d853691297f` |
| `exp1_a_s0_r5` | complete (15:22, 31 min) | 0.045532 | 0.024193 | 3.682e-02 | 308.1 | `28e7fb938505a9f4` |

**σ(mAP50) = 0.00248973, mean 0.044733 (5.57% relative), range 0.007485.**
**σ(mAP50-95) = 0.00143954, mean 0.023672 (6.08% relative).**
**DISTINCT WEIGHT HASHES: 6 out of 6.**

The distinct hash count is the primary evidence, because it is the one quantity that cannot be explained
as weight decay or as a metric that failed to resolve: six identical configurations at seed 0, same data
order, same effective batch, produced six different parameter digests. `rel_weight_delta` is
0.0362–0.0380 in every run with the < 1e-05 warning silent, and `epochs`/`optimizer_steps` are identical
(6 / 2424) across all six, so this is a training model measured under a fixed budget.

For scale: the frozen-model grid earlier in this experiment gave σ(mAP50) = 4.7e-08, i.e. **~53000×
smaller**, and that number was an artefact of weight decay rather than a noise floor. The whole
difference between the two is whether the model was training (see §5).

**This is the §1 decision rule's first branch, not the "arm (a) shows σ = 0" branch.** Arm (a) exhibits
non-zero same-seed jitter on the second architecture, so there is jitter for the gather sampler to
remove, and the a-versus-d contrast is meaningful.

### ETA and progress

Arm (c) started 15:22:31 (3 repeats × ~1.98 h → ≈21:20). Arm (d) then 2 repeats × ~2.86 h → **finish
≈03:05 on 29 Sep, ~2.9 h margin** against the 06:00 auto-shutdown. Per-run wall times are the grid's
own measurements (31 min per arm-(a) run, 304–311 s/epoch), not extrapolations. `RUN_STOP_AT=2026-09-29
05:15` makes the launcher refuse to start a run after that time, so the grid stops cleanly and harvests
rather than being caught mid-run by the shutdown. `exp1_analyse.py` runs after every harvested run and
writes `exp1_sigma.json`, which is itself harvested.

## 11. Blockers / caveats

* **The card is shared.** While the parent's `exp2c_intra` sweep was running it held 23.7 GiB, leaving
  8.3 GiB; the gather arms cannot start in that window (OOM at batch 1) and arm (a) is limited to
  batch 2. The launcher waits rather than racing. Memory contention does not change the numerics, only
  the achievable batch size, which is recorded per run.
* **The grid is sequential on purpose.** Running arm repeats concurrently would halve the wall time but
  would put environment-dependent contention inside the arm whose whole claim is bit-identity. That is
  the one thing the experiment must not do.
* **Cost.** Measured: arm a 5.06 min/epoch, arm c 19.83, arm d 57.24. The launched budget is 14.8 h,
  with arm (d) 5.7 h of it. The naive uniform-10-epoch grid would have been ≈35 h; the reduction comes
  from shortening arm (d) only, where the claim does not depend on training progress.
* **Hard shutdown at 06:00.** The server auto-shuts-down on 29 Sep at 06:00, which is what forced the
  arm-(d) budget from 3 repeats to 2 and what the `RUN_STOP_AT` guard exists for.
* **Disk.** Each run writes ~165 MB of weights plus a small JSON; the whole grid is ≈1.5 GB against
  16 GB free. No per-epoch checkpoints are written, by design. Weights stay remote — the JSON carries
  the sha256 and that is the bit-identity evidence.
* **Evaluation approximation.** COCO metrics come from `pycocotools` with `maxDets=100` and the COCO
  area bins (small < 32², medium 32²–96², large > 96²). Boxes are restored to source-image pixels
  before evaluation. Size-bin AP is `-1` (undefined) whenever a bin holds no ground truth — which is
  the case for every bin except `all` on VisDrone at this accuracy, because almost no prediction
  reaches the medium/large bins. The aggregator excludes those sentinels instead of averaging them, so
  per-size σ is **undefined, not zero** (see §0, the declared scope boundary).
* **Absolute accuracy is low** (mAP50 ≈ 0.045). That is expected: 6 epochs, and the 91-way COCO head
  reinitialised to 10 classes. The outcome of record is σ across same-seed repeats, not accuracy.


