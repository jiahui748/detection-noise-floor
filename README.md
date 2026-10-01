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
samplers/     the deterministic sampling operators (see samplers/README.md)
harness/      every training script and launcher that produced the results
analysis/     the scripts that turn the measurements into the tables and figures,
              plus the checks that verify the numbers in the manuscript
```

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
python analysis/make_tables.py        # tables from the probe results
python analysis/make_exp2b_assets.py  # the dose-response table and figure
python analysis/gen_survey_macros.py  # survey macros from the survey CSV
```

The training harnesses in `harness/` are the ones actually used; their paths refer to a
remote GPU box and a benchmark layout, so they are included as a record of the protocol
rather than as a turnkey script.

## License

MIT (see `LICENSE`).

## Citation

If you use the sampler, please cite the paper. A DOI will be added here
once the release is archived.
