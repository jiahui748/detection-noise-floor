# detection-noise-floor

Measurement corpus, per-run probe results and reference implementations of two
deformable-attention sampling operators, supporting the paper

> **Reliability of empirical claims in small-object pattern recognition**
> Ma Wei, Jiahui He, Rong Zhang

The paper measures how much of the reported run-to-run variation in small-object
detection is measurement noise rather than method improvement, traces the cause to the
backward accumulation order of the deformable-attention sampling operator, and provides
a deterministic replacement.

## Layout

```
corpus/       measurement corpora and derived quantities
probes/       90 per-run probe results: the raw measurements behind the tables
samplers/     the deterministic sampling operators (see samplers/README.md)
harness/      every training script and launcher that produced the corpora
analysis/     the scripts that turn probes into the paper's tables and figures,
              plus the checks that verify the numbers in the manuscript
notes/        the pre-registration, and the notes from the second-architecture run
```

## Where each paper artefact comes from

| Paper artefact | File in this repository |
|---|---|
| Table 1, per-size noise floor | `probes/size_probe_*.json` (90 runs), reduced by `analysis/make_tables.py` |
| Table A1, the measurement corpus | `probes/` for the enumerated runs |
| Table 12 / Figure 1, full-detector perturbation dose response | `corpus/divergence.csv`, `corpus/divergence_meta/` |
| Section 6.3, the early transient rate (1.18-1.21 per step) | `corpus/intra_reduced.json` |
| Section 6.1, operator noise on real activations | `corpus/realact.json` |
| Table 2 and the survey figures | `corpus/survey_visdrone.csv` (the survey; `survey_report.md` is the write-up) |
| Section 5.4, the Deformable-DETR replication | `corpus/exp1_*.json`, `corpus/exp1_sigma.json` |
| The sampler of Section 7 | `samplers/det_deform_attn.py` |

## A note on the survey file

`corpus/survey_visdrone.csv` holds **one row per (paper x metric family)**, not one row
per paper: seven of the fifteen papers contribute a row to each of two metric families,
so the file has 22 rows for 15 papers. Paper counts must be obtained by grouping.
`analysis/gen_survey_macros.py` does this and the paper draws every survey figure from
its output, so the manuscript and this file cannot drift apart.

## The sampler

See `samplers/README.md` for the full account. In short:



`samplers/det_deform_attn.py` replaces `F.grid_sample` with a gather-based bilinear
sampler that has a deterministic backward pass. It is a drop-in replacement and is legal
under `torch.use_deterministic_algorithms(True)`, under which the native op raises
`grid_sampler_2d_backward_cuda does not have a deterministic implementation`. It costs
1.03x end to end and makes same-seed training repeats bit-identical.

One caveat carried over from the paper: the index bound that makes the gather safe is a
proof **for finite coordinates only**. Clamping does not remove a NaN, and `NaN.long()`
is `INT64_MIN`, which sends the gather out of range. The shipped module sanitises the
coordinates at the source rather than the indices.

## Reproducing the numbers

```
python analysis/make_tables.py        # tables from probes/
python analysis/make_exp2b_assets.py  # the dose-response table and figure
python analysis/gen_survey_macros.py  # survey macros from corpus/survey_visdrone.csv
```

The scripts in `analysis/` read from the released data and regenerate the paper's tables and
survey figures. The training harnesses in `harness/` are the ones actually used; their paths refer to a
remote GPU box and a benchmark layout, so they are included as a record of the protocol
rather than as a turnkey script.

## License

Code under `samplers/`, `harness/` and `analysis/`: MIT (see `LICENSE`).
Data under `corpus/` and `probes/`: CC BY 4.0 (see `LICENSE-DATA`).

## Citation

If you use the corpus or the sampler, please cite the paper. A DOI will be added here
once the release is archived.

## A note on the analysis scripts

The scripts in `analysis/` are the ones that produced the manuscript's tables and
figures, so they write **into the manuscript's directory tree** (`paper_PR/tables/`,
`paper_PR/figures/`) rather than next to themselves. If you run them outside that
tree, create the target directory first or edit the `OUT` constant near the top of
each script:

```
python analysis/gen_survey_macros.py    # rewrites the survey macro block of paper_PR/paper_PR.tex
python analysis/make_tables.py          # writes paper_PR/tables/*.tex
python analysis/make_figures.py         # writes paper_PR/figures/*.pdf
python analysis/make_exp2b_assets.py    # writes the dose-response table and figure
python analysis/recompute_survey_authoritative.py   # recomputes the survey statistics
```

`recompute_survey_authoritative.py` is the one to run first if you want to check the
survey numbers: it reads `corpus/survey_visdrone.csv` and prints the family medians,
means and the reporting-practice census without touching any other file.
