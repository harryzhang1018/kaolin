# Simplicits simulation-in-the-loop — progress

Status of `kaolin/experimental/simplicits_diffsim/`, research code for training the Simplicits
neural skinning field `W_θ` *through* its reduced simulator rather than with the data-free
randomized-elastic + orthogonality objective.

Last updated 2026-08-11. Design plan and the full measurement log:
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

65 tests pass in `tests/python/kaolin/experimental/simplicits_diffsim/` (9 of them
`test_multi_scenario.py`). Nothing under `kaolin/physics/**` was modified.

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

> **Read this next to "Scaling the training pool" below.** That 15× is one trajectory, and most of it
> is specialization to it. Trained and evaluated on a pool of 24 loads, the same comparison gives
> **2.07×**. The single-trajectory number is not mis-measured, but it does not survive scaling.

Videos in `data/videos/` (generated, not committed): full-order FEM, data-free, and
simulation-in-the-loop, sharing one camera, with the FEM drawn as a ghost behind the two reduced
runs.

## Scaling the training pool

One trajectory cannot support a claim about a *basis*: a field can fit the single deformation family
one trajectory visits long before it has learned the object. Since `W_θ` maps a point of *this*
object's bounding box to *this* object's handle weights, it is per-object by construction, so scaling
its training data means more loads and more frames — not more geometries.

**Dataset.** 24 scenarios: 12 Fibonacci-sphere gravity directions × magnitudes 4.9 and 9.8 m/s²,
60 frames each at 20×6×6, generated in 189 s. Same geometry, material, pins and `dt`; only the load
differs. Peak displacements span 0.28–1.44 m, so the pool holds genuinely different deformations
(transverse bending, axial stretch, oblique twist) rather than rescalings of one.

Gravity is the cheap axis on purpose: it is a body force, so it changes only the load term and leaves
`B`, `∂F/∂z` and `BᵀMB` untouched. One network evaluation and one QR serve every scenario in a step
(`ReducedModel.with_pt_forces`).

**Pool scaling**, 8 handles (96 DOFs), 2000 steps at *every* pool size, so more data never buys more
compute. Error is a 60-frame free-running rollout from rest:

| pool | windows | trained mean | worst scenario | floor | rollout/floor |
|---|---|---|---|---|---|
| 1 | 39 | 1.41e-3 m | 1.41e-3 | 5.51e-4 | **2.6×** |
| 2 | 78 | 5.27e-2 | 6.99e-2 | 1.81e-3 | 29.2× |
| 4 | 156 | 5.02e-2 | 8.45e-2 | 2.75e-3 | 18.2× |
| 8 | 312 | 3.94e-2 | 9.74e-2 | 1.56e-3 | 25.3× |
| 16 | 624 | 3.72e-2 | 9.18e-2 | 1.34e-3 | 27.9× |
| 24 | 936 | 2.95e-2 | 9.70e-2 | 1.38e-3 | 21.4× |

`stratified_indices(1)` picks a half-magnitude load, the easiest target in the sweep, so a rising
pooled error could be a harder target rather than a worse field. It is not: scenario 0's *own* error,
held fixed across every pool, goes **1.41e-3 → 3.55e-2 when a single second load is added**, then
recovers only to 2.42e-2 at 24. One extra load costs 25× on the original load; scaling to 24 wins
back about a third.

**The bottleneck is neither the data nor the basis.** The floor says the learned basis represents all
24 trajectories to 1.38e-3 m while the rollout achieves 2.95e-2 — a 21× gap. The handle sweep on the
24-scenario pool makes it unambiguous:

| handles | DOFs | rollout | floor | rollout/floor |
|---|---|---|---|---|
| 4 | 48 | 5.33e-2 | 3.10e-3 | 17× |
| 8 | 96 | 2.95e-2 | 1.38e-3 | 21× |
| 16 | 192 | 4.15e-2 | 8.81e-4 | 47× |
| 32 | 384 | 7.40e-2 | 6.02e-4 | **123×** |

Static representation improves monotonically with DOFs (5× better floor); the trained rollout gets
2.5× *worse*. No capacity limit produces that.

### What it is not

Two candidate mechanisms were tested and **ruled out**, one of them the author's leading hypothesis:

* **Unrolled Newton stopping short of the root.** Expected the 4-iteration solve to miss by an amount
  growing with `cond(H)`. Measured on a projection-pretrained basis: relative residual **~1e-13 after
  4 iterations at every handle count**, and a 4-iteration 60-frame rollout matches a 30-iteration one
  to **1e-9 m** even at 384 DOFs where `cond(H) ≈ 5e4`. The forward operator *is* the backward-Euler
  solution; 4 iterations are enough. Hypothesis dead.
* **Undertraining.** 8000 steps at the 24-scenario pool gives 3.26e-2 — *worse* than 2000 steps'
  2.95e-2. The held-out metric plateaus by step ~1000 (6.88e-3) and never improves over the remaining
  7000. 4× compute buys nothing.

### What it is: the loss and the metric ask different questions

Training supervises horizon-4 windows whose initial state is *projected from full-order ground truth*;
the metric is a 60-frame free-running rollout from rest. The field is never trained on the compounding
of its own error. Decomposing one trained field three ways:

| evaluation | mean error | vs floor |
|---|---|---|
| projection floor (best possible) | 1.19e-3 m | 1.0× |
| teacher-forced 1-step | 3.57e-3 | 3.0× |
| teacher-forced 4-step — *what the loss optimizes* | 5.83e-3 | 4.9× |
| free-running 60-step — *what the metric reports* | 1.84e-2 | 15.5× |

Local dynamics are within 3× of the best the basis allows. Free-running error grows monotonically over
the first ~20 frames (3.5e-4 → 3.1e-2) and then oscillates with the beam's own motion rather than
diverging: phase drift, not instability.

Lengthening the training horizon confirms it positively, and at matched compute:

| training horizon | 60-frame rollout | worst scenario | floor | rollout/floor | time |
|---|---|---|---|---|---|
| 4 | 2.95e-2 m | 9.70e-2 | 1.38e-3 | 21.4× | 113 s |
| 8 | 2.37e-2 | 7.52e-2 | 1.11e-3 | 21.3× | 205 s |
| 16 | **1.22e-2** | 3.60e-2 | 1.18e-3 | **10.3×** | 391 s |

The floor barely moves (1.38e-3 → 1.18e-3), so this is not a representation change — it is accumulation,
and the rollout/floor ratio halves. Compute-matched: horizon 16 reaches 1.22e-2 in 391 s where
horizon 4 given *more* time (8000 steps, 495 s) reaches only 3.26e-2 — **2.7× better for less wall
clock**. The training horizon, not the data volume or the handle count, is the live knob.

### Against the data-free baseline, on the same 24 loads

| | 60-frame rollout | floor |
|---|---|---|
| data-free (4000 steps, published recipe) | 6.11e-2 m | 3.88e-3 m |
| simulation-in-the-loop (h=4, 2000 steps) | 2.95e-2 | 1.38e-3 |
| ratio | **2.07×** | 2.82× |

A real but far more modest advantage than the single-trajectory 15×, and the honest headline for the
method as it stands. Mean relative error of the 24-scenario field is 3.20% of each scenario's motion
(1.43%–7.20%), roughly independent of load magnitude (3.36% at 4.9 m/s², 3.03% at 9.8), which is
consistent with accumulation rather than a nonlinearity breakdown.

Reproduce one configuration end to end with:

```bash
python -m kaolin.experimental.simplicits_diffsim.data_gen.gen_fom_beam \
    --resolution 20 6 6 --frames 60 --sweep-directions 12 --sweep-magnitudes 4.9 9.8
python -m kaolin.experimental.simplicits_diffsim.trainer --beam --resolution 20 6 6 --frames 60 \
    --scenarios 12 --magnitudes 4.9 9.8 --handles 8 --horizon 16 --steps 2000 --pretrain 500 \
    --mode both
```

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

5. **A multi-scenario dataset must validate what it shares.** `TrajectoryDataset` originally collapsed
   `controls` to trajectory 0's, and the trainer resolved gravity, material, pins and masses from it
   once — so a pool of differently-loaded scenarios would have trained every one of them against
   trajectory 0's load and reported a plausibly decreasing loss, with nothing raised. Now `gravity` is
   explicitly per-scenario and the controls folded into the quadrature weights and `BᵀMB`
   (`timestep`, material, pin predicate) are checked equal with a named `ValueError`. The regression
   test asserts *physics*, not plumbing: three scenarios start from rest under mutually orthogonal or
   opposed gravity and each must move along its own `−g`. Under the old behaviour the opposed
   scenario moves **−5.2e-2 m** instead of +5.2e-2 — the test fails on the sign, which is why it is
   worth having.

Latent upstream issues found, all left unfixed since this work touches no production code:

6. **`_neohookean_gradient` is wrong away from `F = I`.** It builds the volumetric term with
   `torch.linalg.inv(F)` where `∂J/∂F = J F⁻ᵀ` needs the transpose; substituting
   `inv(F).transpose(-2, -1)` makes it match autograd of its own energy to 2.9e-11. No production
   path calls it (`loss_elastic` uses only `_neohookean_energy`; the simulator uses the correct Warp
   kernel). It is untested because `test_neohookean_gradient`'s fixture leaves `F` exactly the
   identity — its `+ eps * torch.rand(...)` is a separate statement, not a continuation — and
   `I⁻¹ = I⁻ᵀ`. Pinned by
   `test_materials_torch.py::test_gradient_disagrees_with_buggy_upstream_torch_reference`.
7. **`loss_ortho` cannot be backpropagated in float64.** Its identity is built with
   `torch.eye(..., device=...)` and no `dtype`. Worked around by `losses.ortho_term`, asserted equal
   to it in float32.
8. **`NeohookeanElasticMaterial.gradients`** is preallocated as `vec9` while
   `_neohookean_gradient_wp_kernel` declares its output as `mat33`, so passing the preallocated
   buffer raises. The default `wp.zeros_like(defo_grads)` path works.
9. **Training and simulation disagree on Neo-Hookean λ.** `loss_elastic` uses
   `reparameterize_lame=False`; the scene uses `True` (`λ ← λ + μ`). The twin follows the simulator.
10. **`add_object` defaults differ from `SimulatedObject.__init__`** on `normalize_weights_by_samples`
   and `apply_qr` (True/True vs False/False).
11. **5 pre-existing test failures** in `tests/python/kaolin/physics/simplicits/test_simplicits_vs_fem.py`
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
  dataset.py          trajectory windows, projected initial states, frame-range splits,
                      multi-scenario pooling with validated shared controls
  trainer.py          training loop, curriculum, diagnostics, data-free baseline,
                      per-scenario loads over one shared assembly
  fd_check.py         directional finite-difference gradient gate
  benchmark.py        ms/frame, FPS, real-time factor
  viz.py              trajectory → mp4
  data_gen/fem_hex.py, data_gen/gen_fom_beam.py   full-order ground truth
```

## Not done

- **The training horizon**, now the highest-value knob. Horizon 16 already halves rollout/floor; the
  untried steps are a curriculum past 16 toward the full 60 (`train(horizon_schedule=...)` supports
  it), turning on the velocity loss (`vel_coeff` is 0), and scheduled sampling — rolling windows from
  the field's *own* previous state instead of the projected ground-truth one, which is the direct fix
  for the teacher-forcing gap rather than a mitigation of it.
- **Held-out load cases.** Now well-motivated rather than premature: the pool exists, and the field
  fits it to 3.2% of motion. Note the trained-pool numbers already hint at the answer — a field
  trained on one scenario scores 3.22e-2 on all 24 while one trained on all 24 scores 2.95e-2, i.e.
  scaling the pool 24× bought ~9% on the full set. Worth measuring properly with the accumulation
  problem fixed first, since it currently dominates.
- **Boundary-condition and material variation.** Both are blocked on the same thing: they change
  `BᵀMB` and the quadrature weights, which the trainer resolves once, so they raise by design today.
  Varying pins additionally interacts with the hard-Dirichlet-vs-penalty mismatch below.
- **Applied tractions and non-rest initial conditions.** `FullOrderNeohookeanSolver` has gravity and
  pins only, and `rollout` starts from rest, so the load axis is currently body-force-only.
- **Stage B** (implicit-function / adjoint differentiation of the converged root). The convergence
  measurement above strengthens the case: at 4 iterations the residual is already ~1e-13, so the
  iterate *is* a root and the implicit function theorem applies without qualification — Stage B would
  trade the unrolled graph for a single linear solve at no accuracy cost. Design notes to
  carry forward: the adjoint must use the **unregularized** Jacobian, since `_newton_H` adds
  `newton_hessian_regularizer · I` — which is why `ReducedModel.hessian` excludes it — and the twin
  needs its own `‖g‖`-based convergence test, because the Warp solver's `|Δz·g| < conv_tol` is checked
  *before* stepping and never reports success. `losses_warp._EnergyPotential` is the
  `torch.autograd.Function` template.
- Contact/floor during training. The machinery is verified against Warp with an *active* floor, but
  the barrier is non-smooth at activation.
- Overfitting control — largely *answered* by scaling, and no longer the main problem. On one
  trajectory it degraded 8.6e-3 → 2.5e-2 m by step 200 at lr 1e-4 with the floor rising 2.2e-3 →
  5.0e-3. On the 24-scenario pool it instead plateaus: best 6.88e-3 at step 1000, then 9e-3–1.3e-2 for
  7000 more steps with no upward trend. More data flattened the curve rather than lowering it, which
  is what pointed at the horizon instead.
- Reference-resolution (21³) training runs; only the coarse and 20×6×6 beams were used.
