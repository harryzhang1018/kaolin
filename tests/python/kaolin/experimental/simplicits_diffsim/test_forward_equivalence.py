# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Layers 3-5 of the equivalence ladder: the assembled backward-Euler step.

Layer 3 -- the residual and Hessian at a given reduced state -- is the decisive one. It is what
licenses any later claim that the twin's trajectories mean the same thing as the production
simulator's. Layers 4 and 5 then check one Newton step and a multi-step rollout; the rollout is
reported as *drift*, not exactness, because the production line search is not a plain Armijo
search (see :func:`~kaolin.experimental.simplicits_diffsim.step.armijo_step_size`) and
``SimplicitsScene`` runs in float32 only.
"""

import pytest
import torch
import warp as wp

from kaolin.experimental.simplicits_diffsim.step import newton_step_unrolled


@pytest.mark.parametrize('device', ['cuda'])
def test_residual_matches_warp_at_rest(device, make_matched_pair):
    """At z = 0 the residual is pure external force; it must match ``_newton_G`` exactly."""
    pair = make_matched_pair(device)
    zeros = pair.model.zeros()
    expected = pair.warp_residual(zeros, zeros, zeros)
    actual = pair.model.residual(zeros, zeros, zeros, pair.timestep)
    assert (actual - expected).norm() / expected.norm() < 1e-5


@pytest.mark.parametrize('device', ['cuda'])
@pytest.mark.parametrize('seed', [1, 2, 3])
def test_residual_and_hessian_match_warp(device, seed, make_matched_pair, make_reduced_state):
    """Layer 3: residual and Hessian must match the Warp assembly at random reduced states."""
    pair = make_matched_pair(device)
    coords, prev, velocity = make_reduced_state(pair, seed=seed)

    expected_g = pair.warp_residual(coords, prev, velocity)
    actual_g = pair.model.residual(coords, prev, velocity, pair.timestep)
    assert (actual_g - expected_g).norm() / expected_g.norm() < 1e-5

    expected_h = pair.warp_hessian(coords)
    actual_h = pair.model.hessian(coords, pair.timestep)
    assert (actual_h - expected_h).norm() / expected_h.norm() < 1e-5
    assert (actual_h - actual_h.transpose(0, 1)).abs().max() / actual_h.abs().max() < 1e-6


@pytest.mark.parametrize('device', ['cuda'])
def test_incremental_potential_matches_warp(device, make_matched_pair, make_reduced_state):
    """The line-search objective must match ``_newton_E``, or step sizes will diverge."""
    pair = make_matched_pair(device)
    coords, prev, velocity = make_reduced_state(pair, seed=4)
    expected = pair.warp_energy(coords, prev, velocity)
    actual = float(pair.model.newton_energy(coords, prev, velocity, pair.timestep))
    assert abs(actual - expected) / max(abs(expected), 1e-12) < 1e-4


@pytest.mark.parametrize('device', ['cuda'])
def test_residual_is_gradient_of_incremental_potential(device, make_matched_pair,
                                                       make_reduced_state, make_float64_twin):
    """The analytic residual must be the true gradient of the incremental potential.

    The residual is written in closed form (not by autograd) to keep the differentiated path
    first-order, so this consistency has to be tested rather than assumed.
    """
    pair = make_matched_pair(device, num_qp=64, num_handles=3)
    model = make_float64_twin(pair)
    coords, prev, velocity = make_reduced_state(pair, seed=5)
    coords, prev, velocity = coords.double().requires_grad_(True), prev.double(), velocity.double()

    energy = model.newton_energy(coords, prev, velocity, pair.timestep)
    autograd_g, = torch.autograd.grad(energy, coords)
    analytic_g = model.residual(coords, prev, velocity, pair.timestep)
    assert (analytic_g - autograd_g).abs().max() / autograd_g.abs().max() < 1e-9


@pytest.mark.parametrize('device', ['cuda'])
def test_single_newton_direction_matches_warp(device, make_matched_pair, make_reduced_state):
    """Layer 4: one full Newton step (alpha forced to 1) must match a direct Warp solve.

    Compared as a *backward* error, not a forward one. Penalty boundary conditions
    (:math:`k = 10^4`) against small lumped masses leave the reduced Hessian with a condition
    number around :math:`4 \\times 10^7`, so the ~5e-8 agreement in :math:`H` and :math:`g` that
    layer 3 establishes can still permit a percent-level difference in :math:`\\Delta z` itself --
    that is the conditioning talking, not a modelling difference. The condition-independent claim
    is that the twin's direction solves the *Warp* system, which is what is asserted here, plus its
    consequence in position space.
    """
    pair = make_matched_pair(device)
    coords, prev, velocity = make_reduced_state(pair, seed=6)

    warp_h = pair.warp_hessian(coords).double()
    warp_g = pair.warp_residual(coords, prev, velocity).double()
    torch_h = pair.model.hessian(coords, pair.timestep).double()
    torch_g = pair.model.residual(coords, prev, velocity, pair.timestep).double()

    expected_dz = -torch.linalg.solve(warp_h, warp_g)
    actual_dz = -torch.linalg.solve(torch_h, torch_g)

    backward_error = float((warp_h @ actual_dz + warp_g).norm() / warp_g.norm())
    condition = float(torch.linalg.cond(warp_h))
    forward_error = float((actual_dz - expected_dz).norm() / expected_dz.norm())
    print(f'cond(H) = {condition:.3e}, backward error = {backward_error:.3e}, '
          f'forward error = {forward_error:.3e}')
    assert backward_error < 1e-5

    rest = pair.model.pts.double()
    lbs = pair.model.lbs.double()
    step_expected = (lbs @ expected_dz).reshape(-1, 3)
    step_actual = (lbs @ actual_dz).reshape(-1, 3)
    magnitude = step_expected.norm(dim=-1).max()
    assert (step_actual - step_expected).norm(dim=-1).max() / magnitude < 1e-3
    assert rest.shape == step_actual.shape


@pytest.mark.parametrize('device', ['cuda'])
def test_rollout_tracks_warp_simulation(device, make_matched_pair):
    """Layer 5: a short rollout must track ``run_sim_step`` to within measured drift.

    Not an exactness test. The production solver breaks out of Newton on ``|dz . g| < conv_tol``
    *before* stepping, line-searches with the grow-then-accept quirk, and runs in float32, so a
    tolerance here would be a tolerance on accumulated float32 noise. What must hold is that the
    two trajectories stay close relative to the motion they undergo.
    """
    pair = make_matched_pair(device, max_newton_steps=10)
    scene, model = pair.scene, pair.model
    num_steps = 10

    coords = wp.to_torch(scene.sim_z).clone().flatten()
    velocity = torch.zeros_like(coords)
    worst_relative_drift = 0.0

    for _ in range(num_steps):
        scene.run_sim_step()
        warp_coords = wp.to_torch(scene.sim_z).clone().flatten()

        prev = coords
        coords = newton_step_unrolled(model, prev, velocity, pair.timestep,
                                      num_newton_steps=10, line_search=True, conv_tol=1e-4)
        velocity = (coords - prev) / pair.timestep

        warp_positions = model.pts + (model.lbs @ warp_coords).reshape(-1, 3)
        torch_positions = model.positions(coords)
        displacement = (warp_positions - model.pts).norm(dim=-1).max()
        drift = (torch_positions - warp_positions).norm(dim=-1).max()
        worst_relative_drift = max(worst_relative_drift, float(drift / displacement.clamp(min=1e-9)))

    print(f'worst relative position drift over {num_steps} steps: {worst_relative_drift:.3e}')
    assert worst_relative_drift < 5e-2, (
        f'twin drifted {worst_relative_drift:.3e} relative to the deformation magnitude; '
        'layer 3 should be re-checked before trusting the rollout')


@pytest.mark.parametrize('device', ['cuda'])
def test_active_floor_matches_warp(device, make_matched_pair, make_reduced_state):
    """The floor must match the Warp assembly when it is actually violated.

    In the default scenario the floor sits at -1 while every point is in the unit cube, so the
    penalty never activates and its whole non-zero branch would go untested. Raising the floor into
    the object exercises the one-sided gradient and the jump in the Hessian.
    """
    pair = make_matched_pair(device)
    scene = pair.scene
    # Re-place the floor above most of the points so the penalty is on.
    scene.set_scene_floor(floor_height=0.5, floor_axis=1, floor_penalty=10000.0, flip_floor=False)
    from kaolin.experimental.simplicits_diffsim import forces
    pair.model.pt_forces = [f for f in pair.model.pt_forces if not isinstance(f, forces.Floor)]
    pair.model.pt_forces.append(forces.Floor(0.5, 1, False, 10000.0))

    coords, prev, velocity = make_reduced_state(pair, seed=8)
    active = (pair.model.positions(coords)[:, 1] < 0.5).sum()
    assert int(active) > 0, 'the floor must actually be violated for this test to mean anything'

    expected_g = pair.warp_residual(coords, prev, velocity)
    actual_g = pair.model.residual(coords, prev, velocity, pair.timestep)
    assert (actual_g - expected_g).norm() / expected_g.norm() < 1e-5

    expected_h = pair.warp_hessian(coords)
    actual_h = pair.model.hessian(coords, pair.timestep)
    assert (actual_h - expected_h).norm() / expected_h.norm() < 1e-5


@pytest.mark.parametrize('device', ['cuda'])
def test_rest_state_is_a_fixed_point_without_loads(device, make_matched_pair):
    """With no external forces the rest state must be an exact equilibrium."""
    pair = make_matched_pair(device, with_floor=False, with_pins=False)
    pair.model.pt_forces = []
    zeros = pair.model.zeros()
    assert pair.model.residual(zeros, zeros, zeros, pair.timestep).abs().max() < 1e-9
    stepped = newton_step_unrolled(pair.model, zeros, zeros, pair.timestep, num_newton_steps=3)
    assert stepped.abs().max() < 1e-9
