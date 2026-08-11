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

"""The full-order solver is the ground truth, so its own correctness is load-bearing.

Patch tests (does an exactly-representable deformation give the exactly-right energy) plus
derivative consistency (is the assembled tangent really the derivative of the assembled residual).
If these fail, every reduction error measured against this solver is meaningless.
"""

import pytest
import torch

from kaolin.experimental.simplicits_diffsim.data_gen.fem_hex import (FullOrderNeohookeanSolver,
                                                                    HexGrid)
from kaolin.experimental.simplicits_diffsim.materials_torch import neohookean_energy

BOUNDS_MIN = (0.0, 0.75, 0.75)
BOUNDS_MAX = (1.0, 1.0, 1.0)
YOUNGS = 1e5
POISSON = 0.45
DENSITY = 500.0


def _grid(device, resolution=(4, 2, 2)):
    return HexGrid(BOUNDS_MIN, BOUNDS_MAX, resolution, device=device, dtype=torch.float64)


def _solver(grid, pinned=None, gravity=(0.0, 9.8, 0.0)):
    return FullOrderNeohookeanSolver(grid, YOUNGS, POISSON, DENSITY, gravity=gravity,
                                     timestep=0.05, pinned_nodes=pinned)


@pytest.mark.parametrize('device', ['cuda'])
def test_grid_volume_and_mass(device):
    """Quadrature must integrate the volume exactly, and lumped masses must conserve total mass."""
    grid = _grid(device, (5, 3, 3))
    expected = 1.0 * 0.25 * 0.25
    assert abs(float(grid.total_volume()) - expected) / expected < 1e-12
    assert abs(float(grid.nodal_volumes().sum()) - expected) / expected < 1e-12
    masses = grid.lumped_masses(DENSITY)
    assert abs(float(masses.sum()) - DENSITY * expected) / (DENSITY * expected) < 1e-12
    assert float(masses.min()) > 0.0


@pytest.mark.parametrize('device', ['cuda'])
def test_affine_displacement_gives_exact_defo_grad(device):
    """A linear displacement field must reproduce its own gradient at every quadrature point."""
    grid = _grid(device)
    torch.manual_seed(0)
    linear = 0.05 * torch.randn(3, 3, device=device, dtype=torch.float64)
    displacements = grid.nodes @ linear.transpose(0, 1)

    expected = torch.eye(3, device=device, dtype=torch.float64) + linear
    actual = grid.defo_grads(displacements)
    assert (actual - expected).abs().max() < 1e-12


@pytest.mark.parametrize('device', ['cuda'])
def test_patch_test_uniform_deformation_energy(device):
    """A uniform deformation's total energy must equal density times volume, exactly.

    This is the patch test: it validates the shape-function gradients, the quadrature weights and
    the element assembly together, against a closed form that involves none of them.
    """
    grid = _grid(device, (5, 3, 3))
    solver = _solver(grid, gravity=(0.0, 0.0, 0.0))
    torch.manual_seed(1)
    linear = 0.08 * torch.randn(3, 3, device=device, dtype=torch.float64)
    displacements = grid.nodes @ linear.transpose(0, 1)

    defo_grad = (torch.eye(3, device=device, dtype=torch.float64) + linear).unsqueeze(0)
    unit = torch.ones(1, device=device, dtype=torch.float64)
    density = neohookean_energy(defo_grad, solver.mus[:1], solver.lams[:1], unit)
    expected = float(density) * float(grid.total_volume())

    actual = float(solver.potential_energy(displacements))
    assert abs(actual - expected) / abs(expected) < 1e-12


@pytest.mark.parametrize('device', ['cuda'])
def test_gradient_is_derivative_of_energy(device):
    """The assembled nodal gradient must be autograd of the assembled energy."""
    grid = _grid(device)
    solver = _solver(grid)
    torch.manual_seed(2)
    displacements = (0.01 * torch.randn_like(grid.nodes)).requires_grad_(True)

    autograd_grad, = torch.autograd.grad(solver.potential_energy(displacements), displacements)
    analytic = solver.potential_gradient(displacements)
    assert (analytic - autograd_grad).abs().max() / autograd_grad.abs().max() < 1e-9


@pytest.mark.parametrize('device', ['cuda'])
def test_tangent_is_derivative_of_gradient(device):
    """The assembled tangent must be the true Jacobian of the assembled gradient."""
    grid = _grid(device, (3, 2, 2))
    solver = _solver(grid, gravity=(0.0, 0.0, 0.0))
    torch.manual_seed(3)
    displacements = 0.01 * torch.randn_like(grid.nodes)

    num_dofs = 3 * grid.num_nodes
    rows, cols, vals = solver._tangent_triplets(displacements)
    tangent = torch.zeros(num_dofs, num_dofs, device=device, dtype=torch.float64)
    tangent.index_put_((rows, cols), vals, accumulate=True)
    assert (tangent - tangent.transpose(0, 1)).abs().max() / tangent.abs().max() < 1e-10

    direction = torch.randn_like(displacements)
    direction = direction / direction.norm()
    analytic = (tangent @ direction.reshape(-1)).reshape(-1, 3)

    best = float('inf')
    for eps in (1e-5, 1e-6, 1e-7):
        plus = solver.potential_gradient(displacements + eps * direction)
        minus = solver.potential_gradient(displacements - eps * direction)
        fd = (plus - minus) / (2.0 * eps)
        best = min(best, float((fd - analytic).abs().max() / analytic.abs().max()))
    assert best < 1e-7, f'best relative error over the eps sweep was {best:.3e}'


@pytest.mark.parametrize('device', ['cuda'])
def test_rest_state_is_equilibrium_without_gravity(device):
    """With no loads, the rest state must be an exact fixed point of the step."""
    grid = _grid(device)
    solver = _solver(grid, gravity=(0.0, 0.0, 0.0))
    zeros = torch.zeros_like(grid.nodes)
    assert solver.potential_gradient(zeros).abs().max() < 1e-18

    displacements, velocities = solver.step(zeros, zeros)
    assert displacements.abs().max() < 1e-18 and velocities.abs().max() < 1e-18


@pytest.mark.parametrize('device', ['cuda'])
def test_newton_converges_and_pins_hold(device):
    """A loaded step must converge, keep the pinned nodes fixed, and bend the right way."""
    grid = _grid(device, (6, 2, 2))
    pinned = torch.nonzero(grid.nodes[:, 0] >= 0.98, as_tuple=False).squeeze(1)
    assert pinned.numel() > 0
    solver = _solver(grid, pinned=pinned)

    displacements = torch.zeros_like(grid.nodes)
    velocities = torch.zeros_like(grid.nodes)
    for _ in range(3):
        prev_displacements, prev_velocities = displacements, velocities
        displacements, velocities = solver.step(prev_displacements, prev_velocities,
                                                num_newton_steps=30, tolerance=1e-11)

    # The converged state must satisfy the backward-Euler equation written against the state it
    # was stepped from -- not against itself.
    step = displacements - prev_displacements - solver.timestep * prev_velocities
    residual = (solver.masses.unsqueeze(-1) * step
                + solver.timestep ** 2 * solver.potential_gradient(displacements))
    free_residual = residual.reshape(-1)[solver.free_dofs].norm()
    load = (solver.masses.unsqueeze(-1) * solver.gravity).norm() * solver.timestep ** 2
    assert float(free_residual / load) < 1e-8

    assert displacements[pinned].abs().max() < 1e-18
    # Gravity is +y in the energy, so the beam must fall in -y.
    assert float(displacements[:, 1].min()) < -1e-3
    free_tip = grid.nodes[:, 0] < 0.05
    assert float(displacements[free_tip, 1].mean()) < float(displacements[:, 1].mean())


@pytest.mark.parametrize('device', ['cuda'])
def test_cg_solver_agrees_with_dense(device):
    """The GPU CG path must land on the same step as the exact dense solve.

    CG exists so the reference-resolution beam is tractable on the GPU (a CPU refactorization per
    Newton iteration dominates past a few thousand degrees of freedom), so it has to be held to the
    exact solver's answer on a problem small enough to compute both.
    """
    grid = _grid(device, (5, 2, 2))
    pinned = torch.nonzero(grid.nodes[:, 0] >= 0.98, as_tuple=False).squeeze(1)

    trajectories = {}
    for solver_name in ('dense', 'cg'):
        solver = FullOrderNeohookeanSolver(grid, YOUNGS, POISSON, DENSITY, timestep=0.05,
                                           pinned_nodes=pinned, linear_solver=solver_name)
        displacements = torch.zeros_like(grid.nodes)
        velocities = torch.zeros_like(grid.nodes)
        for _ in range(3):
            displacements, velocities = solver.step(displacements, velocities,
                                                    num_newton_steps=20, tolerance=1e-11)
        trajectories[solver_name] = displacements

    scale = trajectories['dense'].norm(dim=-1).max()
    assert float((trajectories['cg'] - trajectories['dense']).norm(dim=-1).max() / scale) < 1e-8


@pytest.mark.parametrize('device', ['cuda'])
def test_surface_faces_reference_valid_nodes(device):
    """Exported boundary triangles must be non-degenerate and index real nodes."""
    grid = _grid(device, (4, 2, 2))
    faces = grid.surface_faces()
    assert faces.shape[1] == 3
    assert int(faces.min()) >= 0 and int(faces.max()) < grid.num_nodes
    assert (faces[:, 0] != faces[:, 1]).all() and (faces[:, 1] != faces[:, 2]).all()

    corners = grid.nodes[faces]
    normals = torch.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0], dim=-1)
    assert float(normals.norm(dim=-1).min()) > 0.0
