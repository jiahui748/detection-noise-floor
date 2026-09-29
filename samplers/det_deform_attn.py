"""Deterministic multi-scale deformable attention.

WHY
    ultralytics/models/rtdetr/train.py states in its own docstring:
        "F.grid_sample used in RT-DETR does not support the `deterministic=True`
         argument."
    and init_seeds() calls torch.use_deterministic_algorithms(True,
    warn_only=True), which WARNS and then proceeds, so the limitation silently
    costs reproducibility. The nondeterminism is only in grid_sample's BACKWARD
    (CUDA scatters gradients with atomicAdd, whose accumulation order varies);
    the forward is deterministic.

    Measured on VisDrone / RT-DETR-L, batch 8, 30 epochs:
      * across 10 seeds and 4 same-seed replicates, F = Var_jitter/Var_total ~ 1
        (p>0.05): the seed contributes no detectable variance.
      * enabling --deterministic alone cut the spread only ~30%.
      * replacing the sampler drove the spread to 0.00000 on all eight metrics;
        three replicates were bit-identical.

WHAT
    Bilinear sampling expressed as an integer gather plus arithmetic, so autograd
    accumulates gradients through index_select/index_put instead of atomicAdd,
    which makes it legal under strict
    torch.use_deterministic_algorithms(True).

BOUND PROOF (why the flat gather cannot go out of range -- no runtime check)
    x0 = floor(x).clamp(0, W-1), y0 = floor(y).clamp(0, H-1)
    x1 = (floor(x)+1).clamp(0, W-1), y1 = (floor(y)+1).clamp(0, H-1)
    so every index used is in [0, dim-1], hence flat = iy*W + ix is in
    [0, H*W-1]. Out-of-range sampling positions are excluded by the PER-CORNER
    mask, not by the index. The interpolation weight uses the UNCLAMPED
    coordinate (x - floor(x)), which is what keeps zero-padding exact.

IMPLEMENTATION HISTORY (all defects found by tests, none hidden)
    v1  flattened with an unchecked product -> index H*W, tripping
        ScatterGatherKernel.cu:163 in final_eval().
    v2  per-axis index_select had a wrong reshape.
    v3  masked the whole stencil by the base cell instead of per corner:
        forward max|diff| = 3.21 against F.grid_sample (silently WRONG).
    v4  correct numerics (5.7e-06) but carried a redundant GPU assert that
        forced a device sync each call and could fire spuriously.
    v5  tried to drop the assert by clamping before flooring, which changed the
        interpolation weights: wrong again (3.27). Reverted.
    v6  = v4's arithmetic with the assert replaced by the proof above.
        Verified: forward 5.7e-06, grad 1.2e-07, strict-mode backward OK.
    Lesson recorded because it generalises: "it runs without error" is not
    evidence of correctness -- only the numerical comparison is.

COST (a trade-off, not a free win)
    ~3.5x on the sampling op alone; ~1.07x end-to-end when only the sampler is
    swapped, ~1.64x when combined with --deterministic.
"""
import torch


def det_grid_sample_bilinear(value, grid):
    """[N,C,H,W] x [N,Ho,Wo,2] -> [N,C,Ho,Wo].

    Drop-in for F.grid_sample(..., mode='bilinear', padding_mode='zeros',
    align_corners=False) with a deterministic backward.
    """
    N, C, H, W = value.shape
    _, Ho, Wo, _ = grid.shape

    x = (grid[..., 0] + 1) * 0.5 * W - 0.5
    y = (grid[..., 1] + 1) * 0.5 * H - 0.5

    # SANITISE AT THE SOURCE. Two failure modes were measured here, and both are
    # reachable in the deployed fork, whose validator warms up on
    # torch.empty(8,3,640,640) -- uninitialised memory:
    #
    #   1. CRASH. clamp() does not remove NaN: clamp(NaN, 0, W-1) is NaN, and
    #      NaN.long() is INT64_MIN (-9223372036854775808), which sends this module's
    #      gather out of bounds and trips
    #        ScatterGatherKernel.cu: `idx_dim >= 0 && idx_dim < index_size`
    #      One non-finite coordinate in 5000 is enough (verified). The bound argument
    #      in the module docstring therefore held only for FINITE coordinates.
    #   2. SILENT NaN. Masking an out-of-range sample does not remove a NaN that has
    #      already reached the interpolation weights, because 0 * NaN is NaN. So
    #      sanitising only the indices stops the crash but leaves a NaN output.
    #
    # Mapping non-finite coordinates far outside the map fixes both: valid() is then
    # False, the weights are finite, and the sample contributes exactly zero, which is
    # the padding_mode='zeros' semantics. For finite coordinates nan_to_num is the
    # identity, so every measurement already reported is unaffected.
    BIG = float(max(H, W) + 2)
    x = torch.nan_to_num(x, nan=BIG, posinf=BIG, neginf=-BIG)
    y = torch.nan_to_num(y, nan=BIG, posinf=BIG, neginf=-BIG)

    x0f = torch.floor(x)
    y0f = torch.floor(y)

    # clamp ONLY for indexing; the weight below uses the unclamped x/y so that
    # out-of-range samples are removed by the mask rather than distorted.
    x0 = x0f.clamp(0, W - 1)
    y0 = y0f.clamp(0, H - 1)
    x1 = (x0f + 1).clamp(0, W - 1)
    y1 = (y0f + 1).clamp(0, H - 1)

    def gather(ix, iy):
        flat = (iy.long() * W + ix.long()).reshape(N, -1)      # in [0, H*W-1]
        v = value.reshape(N, C, H * W)
        out = torch.gather(v, 2, flat.unsqueeze(1).expand(N, C, flat.shape[1]))
        return out.reshape(N, C, Ho, Wo)

    def valid(ix, iy):
        return ((ix >= 0) & (ix <= W - 1) & (iy >= 0) &
                (iy <= H - 1)).to(value.dtype).unsqueeze(1)

    v00 = gather(x0, y0)
    v01 = gather(x1, y0)
    v10 = gather(x0, y1)
    v11 = gather(x1, y1)

    m00 = valid(x0f, y0f)
    m01 = valid(x0f + 1, y0f)
    m10 = valid(x0f, y0f + 1)
    m11 = valid(x0f + 1, y0f + 1)

    wx1 = (x - x0f).unsqueeze(1)
    wx0 = 1.0 - wx1
    wy1 = (y - y0f).unsqueeze(1)
    wy0 = 1.0 - wy1

    return (v00 * m00 * wx0 * wy0 + v01 * m01 * wx1 * wy0 +
            v10 * m10 * wx0 * wy1 + v11 * m11 * wx1 * wy1)


def multi_scale_deformable_attn_deterministic(value, value_spatial_shapes,
                                              sampling_locations, attention_weights):
    """Same signature/return as ultralytics' multi_scale_deformable_attn_pytorch."""
    bs, _, num_heads, embed_dims = value.shape
    _, num_queries, _, num_levels, num_points, _ = sampling_locations.shape
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for level, (H_, W_) in enumerate(value_spatial_shapes):
        value_l_ = (value_list[level].flatten(2).transpose(1, 2)
                    .reshape(bs * num_heads, embed_dims, H_, W_))
        sampling_grid_l_ = (sampling_grids[:, :, :, level]
                            .transpose(1, 2).flatten(0, 1))
        sampling_value_list.append(
            det_grid_sample_bilinear(value_l_, sampling_grid_l_))
    attention_weights = attention_weights.transpose(1, 2).reshape(
        bs * num_heads, 1, num_queries, num_levels * num_points)
    output = ((torch.stack(sampling_value_list, dim=-2).flatten(-2)
               * attention_weights).sum(-1).view(bs, num_heads * embed_dims,
                                                 num_queries))
    return output.transpose(1, 2).contiguous()


def apply_deterministic_deform_attn(verbose=True):
    """Monkey-patch every module that imported the sampling function by name."""
    import ultralytics.nn.modules.utils as u
    from ultralytics.nn.modules import transformer as tr
    patched = []
    for mod in (u, tr):
        if hasattr(mod, 'multi_scale_deformable_attn_pytorch'):
            mod.multi_scale_deformable_attn_pytorch = \
                multi_scale_deformable_attn_deterministic
            patched.append(mod.__name__)
    try:
        from ultralytics.nn.modules import head as hd
        if hasattr(hd, 'multi_scale_deformable_attn_pytorch'):
            hd.multi_scale_deformable_attn_pytorch = \
                multi_scale_deformable_attn_deterministic
            patched.append(hd.__name__)
    except Exception:
        pass
    if verbose:
        print("[det-attn] deterministic deformable attention applied to:",
              ", ".join(patched) if patched else "NOTHING (check import path)")
    return patched
