# Preregistered plan for the two experiments required before PR submission

**Status: BLOCKED on hardware.** The box is in AutoDL CPU-only mode: `torch.cuda.is_available() == False`,
`torch.cuda.device_count() == 0`, `/dev/nvidia*` absent, `nvidia-smi` permission-denied. Nothing here has been run.
This document pins the design **before** any result is seen, which is the same standard the paper applies to
everyone else. Nothing in it has been added to the manuscript; the paper's claims are unchanged and stay
scoped as they are until these measurements exist.

---

## 0. Environment findings that shape the plan

| finding | consequence |
|---|---|
| No GPU | both experiments need training or full-model backward; neither can run now |
| No `transformers`, no `mmdet`/`mmcv` | DINO and Deformable-DETR are not present; a second architecture must be brought in |
| Direct HTTPS to pypi/github/huggingface **blocked** | cannot `git clone` Deformable-DETR or download HF checkpoints directly |
| `pip` via the Aliyun mirror **works** (`transformers-5.17.0` downloaded, 2.8 MB/s) | `pip install transformers` is viable; that alone provides `DeformableDetrForObjectDetection` and `DinoForObjectDetection` |
| Hugging Face Hub blocked | weights must come from a mirror (`hf-mirror.com`, to be verified) or models must be trained from scratch |
| Trained RT-DETR-L checkpoints present (`result_sdd/*/weights/best.pt`) | Experiment 2 needs **no new weights** |

**Key design consequence.** Hugging Face's `multi_scale_deformable_attention` is implemented with
`F.grid_sample`, i.e. the *same* operator the paper identifies. So a second detector taken from
`transformers` tests the mechanism directly rather than by analogy — no re-implementation of the operator
is needed, and the sampler under test can be dropped into it unchanged.

**Second design consequence.** Experiment 1 does not need competitive AP. The question is whether
run-to-run σ is non-zero and whether the gather sampler removes it, not whether the detector is accurate.
A small backbone and a short schedule therefore suffice, which is what makes this affordable.

---

## 1. Experiment 1 — architecture-level replication

**Claim under test.** *The non-determinism is a property of multi-scale deformable attention, not of
RT-DETR.* The paper currently argues this only at the operator level (§5.4, Table 9, a minimal module) and
on one full architecture (RT-DETR-L). A reviewer can reasonably call the full-model evidence
RT-DETR-specific.

**Design.** Add a second full architecture from a different codebase and family, trained on the same
benchmark with the same protocol.

| item | value |
|---|---|
| detectors | `DeformableDetrForObjectDetection` **and** `DinoForObjectDetection` (transformers), both R50 unless that exceeds memory, in which case ResNet-18 |
| init | from scratch (HF Hub blocked). If `hf-mirror.com` works, prefer COCO-pretrained for both, matching how RT-DETR-L was used |
| benchmark | VisDrone2019-DET, 640 px, the same split and the same probe |
| arms | (a) `F.grid_sample`, strict off; (b) `F.grid_sample`, strict on; (c) gather sampler, strict off; (d) gather sampler, strict on |
| seeds | 5 distinct seeds per arm, **plus 6 same-seed replicates on arm (a)** — the jitter estimate, exactly as for RT-DETR |
| metrics | the same eight: mAP50, mAP50-95, and AP per size bin |
| primary outcome | σ over same-seed replicates, arms (a) vs (d) |

**Pre-registered decision rule.**

* If arm (a) shows σ > 0 and arm (d) shows σ = 0 with bit-identical replicates on either new detector,
  the paper's §2.4/§5.4 claim becomes an architecture-level measurement and the §8.2 limitation
  "amplification was measured on a scaled model" is narrowed to the amplification link only.
* If arm (a) shows σ = 0, the paper **must say so**: the mechanism would be RT-DETR-implementation-specific
  rather than family-wide, and §2.4's list ("Deformable-DETR, DINO, RT-DETR and their descendants") would
  have to be cut back to RT-DETR. This is a real risk and it is why the experiment is worth running.
* If arm (c) shows σ > 0 while (d) shows σ = 0, this replicates the paper's own §7.3 finding (the sampler
  alone does not remove jitter; only the combination does) on a second architecture — a strong result.

**Cost.** 4 arms × (5 + 6) runs = 44 runs per detector. At a short schedule (15–30 epochs, batch 8, 640 px)
and ~2 h per run on a 5090-class card, that is ≈ 90 GPU-hours per detector, ≈ 180 for both. **Run ONE
detector first** (Deformable-DETR, the closer ancestor); add DINO only if the first replicates.

---

## 2. Experiment 2 — real-activation and full-model perturbation evidence

This closes the weakest link in the causal chain. The paper currently reports (i) operator noise on
**synthetic** tensors shaped like a real layer (gradnoise.json) and (ii) amplification in a **standalone
module**, not full RT-DETR (§8.2 limitations 3 and 9, called "the weakest link" in the paper itself).

### 2a. Operator noise on real activations (cheap)

**Design.** Register forward hooks on the deformable-attention modules of the trained RT-DETR-L
(`result_sdd/pn_base_s0/weights/best.pt`), push a **fixed** minibatch of real VisDrone images through in
`eval()` mode, and capture the actual `(value, grid)` tensors entering each MSDeformAttn. Then repeat the
identical backward 8 times on those captured tensors and measure the spread, exactly as in `fig_mech.py`,
and compare against the synthetic magnitudes already reported.

**Arms.** synthetic tensors (existing) vs captured real activations, per layer depth (encoder levels 1–3,
decoder layers 1 and 4) and per tensor shape.

**Pre-registered decision rule.**
* If real-activation relative noise is within ~3× of the synthetic figure, §6.1 stands and limitation 9 is
  discharged with one sentence.
* If it is materially **larger**, every downstream number in §6.3 shifts and the chain must be re-anchored.
* If it is materially **smaller** (e.g. ≪10⁻⁹), the amplification argument weakens and the paper must say
  the synthetic measurement overstated the injected noise.

**Cost.** Forward passes only, no training; runnable on CPU if necessary (a few minutes per image).

### 2b. Full-model perturbation divergence (the strong version)

**Design.** Inject a known relative perturbation ε into the trained RT-DETR-L's parameters, then run a
short identical training segment and track divergence of the **trajectory** and of the **final metric**,
against an unperturbed control.

| item | value |
|---|---|
| ε doses | 0 (control), 10⁻⁹, 10⁻⁸, 10⁻⁶ — log-spaced around the measured operator magnitude |
| segment length | 30–100 optimiser steps on a fixed minibatch order, 3 seeds per dose |
| measurements | ‖δ‖/‖θ‖ vs step; and the resulting mAP50 spread at the end of the segment |
| control | same driver, same batch order, ε = 0, repeated — so the control's own σ is the baseline |

**Pre-registered decision rule.** The paper claims a 10⁻⁹ perturbation "reaches observable scale" and that
fifteen runs land with a 0.3-point mAP50 σ. The test is whether a 10⁻⁹ injection into the **full** detector
produces a final-metric spread of the same order as the observed 0.3 points.

* If yes, the chain from accumulation order to mAP is measured end to end in the real model, and
  limitations 3 and 9 are both discharged. This is the single most valuable experiment in this plan.
* If the injected 10⁻⁹ dose produces no measurable divergence while the 10⁻⁶ dose does, the honest
  conclusion is that the amplification link is real but that the *magnitude* of the synthetic operator
  noise is not sufficient to explain 0.3 points on its own — which would have to be written into §8.2.

**Cost.** 4 doses × 3 seeds = 12 short segments. At ~30 steps and batch 8 this is minutes per segment on a
GPU — far cheaper than a full 30-epoch run. **This is the experiment to run first of all.**

---

## 3. Ordering, given a card

1. **2b** — cheapest, closes the most-cited weakness, and needs no new weights or code dependencies.
2. **2a** — near-free, and it calibrates whether 2b's doses bracket the real operator noise.
3. **1** — one detector only (Deformable-DETR), 44 runs; add DINO on replication.

## 4. What each outcome does to the manuscript

| outcome | manuscript change |
|---|---|
| 2b succeeds | §6.2/§6.3 rewritten around real full-model divergence; limitations 3 and 9 deleted |
| 2b fails at 10⁻⁹ | new limitation: the measured operator noise alone does not reproduce the observed spread; §6.3's last arrow explicitly weakened |
| 1 replicates | §2.4's family claim is earned; title/abstract may name "deformable-attention detectors" rather than RT-DETR |
| 1 does not replicate | §2.4 claim cut back to RT-DETR; the paper becomes narrower and must say so in the abstract |

## 5. Prerequisites to verify before launching

```bash
# 1. a card is actually attached
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
# 2. transformers installs from the mirror
/home/.../python -m pip install -i http://mirrors.aliyun.com/pypi/simple/ transformers
# 3. whether a weights mirror is reachable (otherwise train from scratch)
curl -sI https://hf-mirror.com | head -1
# 4. confirm HF's MSDeformAttn really is grid_sample based, in the installed version
python - <<'EOF'
import inspect, transformers.models.deformable_detr.modeling_deformable_detr as m
src = inspect.getsource(m.multi_scale_deformable_attention)
print('grid_sample' in src, 'deterministic' in src)
EOF
```

Step 4 matters: the whole replication argument rests on the second implementation using the same operator.
If it does not, experiment 1 tests something else and must be redesigned.
