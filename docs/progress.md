# Simplicits simulation-in-the-loop — progress

Status of `kaolin/experimental/simplicits_diffsim/`, research code for training the Simplicits
neural skinning field `W_θ` *through* its reduced simulator rather than with the data-free
randomized-elastic + orthogonality objective.

Last updated 2026-08-10. Design plan and the full measurement log:
`~/.claude/plans/take-a-look-at-hidden-lemon.md`.

## Why a new forward path was needed

Trajectory training needs an unbroken gradient

```
θ → (W, ∇ₓW) → (B, ∂F/∂z, BᵀMB) → {z_t} → {x̂_t} → L
```

The production simulator cannot provide it. At v0.18 it is entirely NVIDIA Warp: `SimulatedObject`
builds `B` and `∂F/∂z` as `warp.sparse` BSR through `wp.from_torch` (severing the graph), the
energy/gradient/Hessian assembly runs Warp kernels with `adjoint=False`, and `newtons_method`,
`_line_search` and `_apply_bounds` are all `@torch.no_grad()`. Removing a few `.detach()` calls —
which would have worked at v0.16 — is not the change.

So this package is a **pure-PyTorch differentiable twin** of that solver, validated layer by layer
against it. The Warp path stays the fast forward path, the large-scale evaluator, and the
correctness oracle.

## Verification

**Forward equivalence against the live Warp path** (`test_forward_equivalence.py`,
`test_kinematics.py`, `test_materials_torch.py`):

| Layer | Compared against | Agreement |
|---|---|---|
| 1 kinematics | `sparse_dFdz_matrix`, `B_dense`, `sim_BMB` | rel < 1e-6 |
| 2 material | `NeohookeanElasticMaterial.{energy,gradient,hessian}` | rel < 1e-5 |
| 3 **assembled step** | `_newton_G` / `_newton_H` at random `z` | **4.6e-8** (H), **3.5e-7** (g) |
| 4 one Newton step | direct Warp solve | backward error **6.0e-7** |
| 5 rollout, 10 steps | `run_sim_step` | drift **4.4e-3** relative |

Layer 3 is the decisive one. Layer 4 is asserted as a *backward* error on purpose: penalty boundary
conditions against small lumped masses leave `cond(H) ≈ 4.1e7`, so `Δz` can differ by ~5e-2 while
`H` and `g` agree to 5e-8. That is conditioning, not a modelling difference.

**Parameter gradients vs finite differences** (`test_gradients.py`, float64, best relative error
over `eps ∈ {1e-4, 1e-5, 1e-6}`):

| Path | rel err |
|---|---|
| 1-step rollout, with and without line search | **1.3e-9** |
| 2-step rollout | 1.5e-7 |
| QR-reparameterized rollout (trainer default) | 8.8e-9 |
| snapshot projection through its normal-equation solve | 6.9e-8 |
| decode at non-quadrature points | passes |

`qr_reparameterization` drops `∂K/∂θ`, which is only sound because the rollout is affine-invariant
in `z`. The FD check above plus a direct invariance test (positions agree to < 1e-8 with and without
QR) is what backs that up.

**Full-order ground truth** (`data_gen/`) reproduces the repo's stored beam reference: `v0` chamfer
**0.0**, `v1` **2.6e-7**, `v_end` (frame 100) **7.4e-5**, against `test_simplicits_vs_fem.py`'s
limits of 1.1e-4 / 1e-2.

56 tests pass in `tests/python/kaolin/experimental/simplicits_diffsim/`. Nothing under
`kaolin/physics/**` was modified.

## Accuracy result

Beam `[0,1] × [0.75,1] × [0.75,1]`, 20×6×6 hex FEM ground truth, 60 frames at `dt = 0.05`. Both
reduced runs share seed, initialization, 8 handles (96 reduced DOFs) and 512 quadrature points — the
only difference is the training objective. Trained on windows inside frames 0–42; frames 43–60 are
extrapolation in time.

| | full-rollout mean | max | held-out frames |
|---|---|---|---|
| data-free (4000 steps, published recipe) | 8.43e-2 m | 4.57e-1 m | 8.04e-2 m |
| **simulation-in-the-loop** (250 projection + 250 rollout steps) | **5.46e-3 m** | 3.19e-2 m | **6.43e-3 m** |
| motion scale | ~1 m | | |

**15× lower** mean error at equal handle count. On the coarser 10×3×3 setup the same comparison gives
8.55e-3 m vs 3.74e-2 m (4.4×), and there the learned basis also improved at pure *representation* —
its best-projection floor dropped from 8.77e-3 to 3.29e-3 m — so it is not simply trading generality
for fit.

Videos in `data/videos/` (generated, not committed): full-order FEM, data-free, and
simulation-in-the-loop, sharing one camera, with the FEM drawn as a ghost behind the two reduced
runs.

## Performance

Same GPU, `dt = 0.05`, CUDA-synchronized, warmup excluded. RTF = simulated seconds per wall-clock
second.

| Simulation | DOFs | ms/frame | sim FPS | RTF |
|---|---|---|---|---|
| Full-order FEM, dense solve | 2940 | 120.2 | 8.3 | 0.42 |
| Full-order FEM, CG solve | 2940 | 87.8 | 11.4 | 0.57 |
| Reduced — data-free | 96 | 6.15 | 162.6 | 8.13 |
| Reduced — sim-in-the-loop | 96 | 6.16 | 162.5 | 8.12 |

The two reduced models cost the **same** to within noise — training changes accuracy, not runtime
cost — so the accuracy gain is free at inference. Reduced cost is also independent of full-order
resolution, which is the point of reduction:

| Full-order mesh | FOM DOFs | FOM ms/frame | FOM RTF | Reduced ms/frame | Speedup |
|---|---|---|---|---|---|
| 20×6×6 | 2940 | 87.8 | 0.57 | 6.15 | 14× |
| 20×20×20 | 26460 | 444.6 | 0.11 | 6.15 | **72×** |

Decoding to mesh vertices adds 0.008 ms (1029 nodes) / 0.023 ms (9261 nodes).

Two honesty notes. The production Warp path measures 7.88 ms/frame here — *slower* than the twin's
5.86 ms in fp32 — but that is not a claim that research code beats production: at 96 DOFs and 512
quadrature points neither is FLOP-bound, and the Warp path pays a device→host sync per energy
evaluation (`_scene_energy.numpy()`), which the line search hits repeatedly. Expect that ordering to
reverse at larger handle counts, more quadrature points, or with collisions. And the full-order
solver is research-grade PyTorch FEM (float64, dense or Jacobi-CG), so treat the speedups as a
comparison between comparable implementations, not a claim against optimized FEM.

## Findings

Design decisions forced by measurement:

1. **`torch.func.hessian` is unsafe in a differentiated path.** Its values are correct, but
   backpropagating *through* it gives a gradient ~24% off (the `jacfwd` outer layer is the culprit);
   `vmap(jacrev(jacrev(·)))` returns NaN. Hence the analytic Neo-Hookean Hessian in
   `materials_torch.py`, verified against the Warp kernel to 1.9e-7 and third-derivative-correct to
   1.1e-9. An AST check in `test_gradients.py` enforces that it stays out.
2. **Never wrap a loss in `torch.no_grad()` inside a finite-difference harness.** It silently changes
   what functorch transforms return, which once produced a confidently wrong "reference".
3. **Warp comparisons are float32-limited** — `SimplicitsScene` hardcodes `self.dtype =
   torch.float32` — so FD gates run in float64 against the twin instead.
4. **The full-order solver is pure PyTorch, not `warp.fem`** (a deliberate deviation from the plan).
   It calls the *same* material functions as the twin, so material mismatch is impossible by
   construction, and those functions are pinned to the Warp kernels by layer 2.

Latent upstream issues found, all left unfixed since this work touches no production code:

5. **`_neohookean_gradient` is wrong away from `F = I`.** It builds the volumetric term with
   `torch.linalg.inv(F)` where `∂J/∂F = J F⁻ᵀ` needs the transpose; substituting
   `inv(F).transpose(-2, -1)` makes it match autograd of its own energy to 2.9e-11. No production
   path calls it (`loss_elastic` uses only `_neohookean_energy`; the simulator uses the correct Warp
   kernel). It is untested because `test_neohookean_gradient`'s fixture leaves `F` exactly the
   identity — its `+ eps * torch.rand(...)` is a separate statement, not a continuation — and
   `I⁻¹ = I⁻ᵀ`. Pinned by
   `test_materials_torch.py::test_gradient_disagrees_with_buggy_upstream_torch_reference`.
6. **`loss_ortho` cannot be backpropagated in float64.** Its identity is built with
   `torch.eye(..., device=...)` and no `dtype`. Worked around by `losses.ortho_term`, asserted equal
   to it in float32.
7. **`NeohookeanElasticMaterial.gradients`** is preallocated as `vec9` while
   `_neohookean_gradient_wp_kernel` declares its output as `mat33`, so passing the preallocated
   buffer raises. The default `wp.zeros_like(defo_grads)` path works.
8. **Training and simulation disagree on Neo-Hookean λ.** `loss_elastic` uses
   `reparameterize_lame=False`; the scene uses `True` (`λ ← λ + μ`). The twin follows the simulator.
9. **`add_object` defaults differ from `SimulatedObject.__init__`** on `normalize_weights_by_samples`
   and `apply_qr` (True/True vs False/False).
10. **5 pre-existing test failures** in `tests/python/kaolin/physics/simplicits/test_simplicits_vs_fem.py`
    on this branch, present before any of this work; 237/242 physics tests pass.

## Layout

```
kaolin/experimental/simplicits_diffsim/
  kinematics.py       dense differentiable B, ∂F/∂z, BᵀMB, QR reparameterization
  materials_torch.py  Neo-Hookean energy / gradient / analytic 9×9 Hessian
  forces.py           torch Gravity / Floor / Boundary
  reduced_model.py    ReducedModel: assembly, potential, residual, Hessian
  step.py             Stage A — unrolled differentiable Newton
  rollout.py          multi-step rollout, truncated BPTT, decode at mesh nodes
  losses.py           position / velocity / defo-grad losses + data-free regularizer
  projection.py       best-fit reduced coords, projection-error floor, POD baseline
  dataset.py          trajectory windows, projected initial states, frame-range splits
  trainer.py          training loop, curriculum, diagnostics, data-free baseline
  fd_check.py         directional finite-difference gradient gate
  benchmark.py        ms/frame, FPS, real-time factor
  viz.py              trajectory → mp4
  data_gen/fem_hex.py, data_gen/gen_fom_beam.py   full-order ground truth
```

## Not done

- **Task-distribution generalization** — multiple loads and boundary conditions per object, with
  held-out *scenarios* rather than held-out frames. This is the experiment that would actually test
  the research claim; everything so far is one trajectory, one seed.
- **Stage B** (implicit-function / adjoint differentiation of the converged root). Design notes to
  carry forward: the adjoint must use the **unregularized** Jacobian, since `_newton_H` adds
  `newton_hessian_regularizer · I` — which is why `ReducedModel.hessian` excludes it — and the twin
  needs its own `‖g‖`-based convergence test, because the Warp solver's `|Δz·g| < conv_tol` is checked
  *before* stepping and never reports success. `losses_warp._EnergyPotential` is the
  `torch.autograd.Function` template.
- Contact/floor during training. The machinery is verified against Warp with an *active* floor, but
  the barrier is non-smooth at activation.
- Overfitting control. Sim-in-the-loop degrades on held-out frames if trained too long at lr 1e-4
  (8.6e-3 → 2.5e-2 m by step 200 on the coarse setup, with the projection floor rising 2.2e-3 →
  5.0e-3). More windows, lower learning rate, or the hybrid regularizer are the next knobs.
- Reference-resolution (21³) training runs; only the coarse and 20×6×6 beams were used.
