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

"""The gradient gate: parameter gradients through the unrolled solver must be correct.

Every layer of the forward equivalence ladder can pass while the *gradient* is quietly wrong --
that is exactly what happens if the Hessian is built with ``torch.func.hessian``. So the
trajectory loss is differentiated with respect to the network parameters and compared against a
central difference of the same loss, in float64.
"""

import pytest
import torch

from kaolin.physics.simplicits.network import SimplicitsMLP
from kaolin.experimental.simplicits_diffsim.fd_check import directional_fd_check
from kaolin.experimental.simplicits_diffsim.kinematics import uniform_sample_volumes
from kaolin.experimental.simplicits_diffsim.reduced_model import build_reduced_model
from kaolin.experimental.simplicits_diffsim.rollout import SkinningDecoder, rollout
from kaolin.experimental.simplicits_diffsim import forces

TIMESTEP = 0.05


def _tiny_setup(device, num_handles=3, num_samples=64, seed=0, with_floor=False, use_qr=False):
    """A deliberately tiny float64 problem: FD needs precision, not scale."""
    torch.manual_seed(seed)
    dtype = torch.float64
    skinning_mod = SimplicitsMLP(3, 8, num_handles, 1, bb_min=torch.zeros(3),
                                 bb_max=torch.ones(3)).to(device=device, dtype=dtype)
    pts = torch.rand(num_samples, 3, device=device, dtype=dtype)
    rhos = torch.full((num_samples,), 500.0, device=device, dtype=dtype)
    yms = torch.full((num_samples,), 1e5, device=device, dtype=dtype)
    prs = torch.full((num_samples,), 0.45, device=device, dtype=dtype)
    vols = uniform_sample_volumes(num_samples, 1.0, device=device, dtype=dtype)
    masses = rhos * vols

    from kaolin.physics.materials.material_utils import to_lame
    mus, lams = to_lame(yms, prs)

    pinned = torch.nonzero(pts[:, 0] >= 0.8, as_tuple=False).squeeze(1)
    pt_forces = [forces.Gravity(torch.tensor([0.0, 9.8, 0.0], device=device, dtype=dtype),
                                rhos, vols),
                 forces.Boundary(pinned, pts[pinned].clone(), 1e4)]
    if with_floor:
        pt_forces.append(forces.Floor(-1.0, 1, False, 1e4))

    def build():
        return build_reduced_model(skinning_mod, pts, mus, lams, vols, masses,
                                   pt_forces=pt_forces,
                                   qr_transform='auto' if use_qr else None)

    targets = pts + 0.01 * torch.randn(4, num_samples, 3, device=device, dtype=dtype)
    return skinning_mod, pts, build, targets


@pytest.mark.parametrize('device', ['cuda'])
@pytest.mark.parametrize('horizon', [1, 2])
@pytest.mark.parametrize('line_search', [False, True])
def test_rollout_parameter_gradient_matches_finite_differences(device, horizon, line_search):
    """The plan's Eq. 22 gate: analytic dL/dtheta must match a central difference of L."""
    skinning_mod, pts, build, targets = _tiny_setup(device)

    def loss_fn():
        model = build()
        coords_traj, _ = rollout(model, TIMESTEP, horizon, num_newton_steps=4,
                                 line_search=line_search)
        predicted = torch.stack([model.positions(z) for z in coords_traj], dim=0)
        return ((predicted - targets[:horizon]) ** 2).sum()

    best, analytic, _ = directional_fd_check(loss_fn, skinning_mod.parameters(),
                                             eps_sweep=(1e-4, 1e-5, 1e-6), verbose=True)
    assert abs(analytic) > 0.0
    assert best < 1e-6, f'best relative error over the eps sweep was {best:.3e}'


@pytest.mark.parametrize('device', ['cuda'])
def test_qr_reparameterized_gradient_matches_finite_differences(device):
    """With the QR reparameterization on -- the trainer's default -- the gradient must still be right.

    ``qr_reparameterization`` computes :math:`K` from :math:`B_\\theta` under ``no_grad``, so
    :math:`\\partial K / \\partial \\theta` is dropped. That is only sound because the rollout is
    affine invariant in the reduced coordinates (exact Newton solves, and an Armijo test on
    :math:`g \\cdot d`, are unchanged by :math:`z = K z'`), which makes the loss genuinely
    independent of :math:`K`. This test is what backs that argument up: if the dropped term
    mattered, the finite difference -- which sees :math:`K` move with :math:`\\theta` -- would
    disagree.
    """
    skinning_mod, pts, build, targets = _tiny_setup(device, use_qr=True)

    def loss_fn():
        model = build()
        coords_traj, _ = rollout(model, TIMESTEP, 2, num_newton_steps=4, line_search=True)
        predicted = torch.stack([model.positions(z) for z in coords_traj], dim=0)
        return ((predicted - targets[:2]) ** 2).sum()

    best, analytic, _ = directional_fd_check(loss_fn, skinning_mod.parameters(), verbose=True)
    assert abs(analytic) > 0.0
    assert best < 1e-6, f'best relative error over the eps sweep was {best:.3e}'


@pytest.mark.parametrize('device', ['cuda'])
def test_qr_reparameterization_leaves_the_rollout_invariant(device):
    """A rollout with and without the QR change of variables must decode to the same positions."""
    skinning_mod, pts, build_plain, targets = _tiny_setup(device, use_qr=False)
    _, _, build_qr, _ = _tiny_setup(device, use_qr=True)

    def positions(build):
        model = build()
        coords_traj, _ = rollout(model, TIMESTEP, 3, num_newton_steps=6, line_search=False)
        return torch.stack([model.positions(z) for z in coords_traj], dim=0)

    plain, reparameterized = positions(build_plain), positions(build_qr)
    scale = (plain - pts).norm(dim=-1).max()
    assert (plain - reparameterized).norm(dim=-1).max() / scale < 1e-8


@pytest.mark.parametrize('device', ['cuda'])
def test_decoder_parameter_gradient_matches_finite_differences(device):
    """Gradients must also flow through the decode at points that are not quadrature points."""
    skinning_mod, pts, build, targets = _tiny_setup(device)
    torch.manual_seed(11)
    decode_pts = torch.rand(32, 3, device=device, dtype=torch.float64)
    decode_targets = decode_pts + 0.01 * torch.randn(2, 32, 3, device=device, dtype=torch.float64)

    def loss_fn():
        model = build()
        coords_traj, _ = rollout(model, TIMESTEP, 2, num_newton_steps=3, line_search=False)
        decoder = SkinningDecoder(skinning_mod, decode_pts)
        return ((decoder.trajectory(coords_traj) - decode_targets) ** 2).sum()

    best, analytic, _ = directional_fd_check(loss_fn, skinning_mod.parameters(), verbose=True)
    assert abs(analytic) > 0.0
    assert best < 1e-6, f'best relative error over the eps sweep was {best:.3e}'


@pytest.mark.parametrize('device', ['cuda'])
def test_gradient_survives_truncated_bptt(device):
    """Truncated BPTT must produce a finite, nonzero gradient -- and a different one.

    Detaching at window boundaries is a deliberate approximation, so this checks that the
    machinery works rather than that the value is exact: a truncated gradient that happened to
    equal the full one would mean the detach did nothing.
    """
    skinning_mod, pts, build, targets = _tiny_setup(device)

    def grad_for(bptt_window):
        model = build()
        coords_traj, _ = rollout(model, TIMESTEP, 4, num_newton_steps=3, line_search=False,
                                 bptt_window=bptt_window)
        predicted = torch.stack([model.positions(z) for z in coords_traj], dim=0)
        loss = ((predicted - targets) ** 2).sum()
        grads = torch.autograd.grad(loss, list(skinning_mod.parameters()))
        return torch.cat([g.reshape(-1) for g in grads])

    full = grad_for(None)
    truncated = grad_for(1)
    assert torch.isfinite(full).all() and torch.isfinite(truncated).all()
    assert full.norm() > 0 and truncated.norm() > 0
    assert (full - truncated).norm() / full.norm() > 1e-8


@pytest.mark.parametrize('device', ['cuda'])
def test_projection_parameter_gradient_matches_finite_differences(device):
    """The snapshot-projection objective must be differentiable through its normal-equation solve."""
    from kaolin.experimental.simplicits_diffsim.kinematics import dense_lbs_matrix
    from kaolin.experimental.simplicits_diffsim.projection import projection_error

    skinning_mod, pts, _, targets = _tiny_setup(device)
    displacements = targets - pts

    def loss_fn():
        lbs = dense_lbs_matrix(pts, skinning_mod.compute_skinning_weights(pts))
        error, _, _ = projection_error(lbs, displacements, ridge=1e-10)
        return error

    best, analytic, _ = directional_fd_check(loss_fn, skinning_mod.parameters(), verbose=True)
    assert abs(analytic) > 0.0
    assert best < 1e-6, f'best relative error over the eps sweep was {best:.3e}'


@pytest.mark.parametrize('device', ['cuda'])
def test_no_functorch_hessian_in_differentiated_path(device):
    """Guard the rule as a test, not a comment: no ``torch.func.hessian`` in Stage A.

    ``jacfwd(jacrev(.))`` yields correct Hessian values but an incorrect derivative, so any
    reintroduction of it into these modules would pass the forward equivalence ladder and fail
    only as a subtly wrong training gradient.
    """
    import ast
    import inspect
    from kaolin.experimental.simplicits_diffsim import (forces as forces_mod, kinematics,
                                                        materials_torch, reduced_model, rollout
                                                        as rollout_mod, step)

    # An AST check rather than a substring search: these modules discuss the banned transforms in
    # their docstrings, and ``ReducedModel.hessian`` is a legitimate call whose attribute is also
    # named ``hessian``. ``jacrev`` is deliberately *not* banned -- ``jacrev(jacrev(.))`` is
    # correct, and ``SkinningModule.compute_dwdx`` relies on it.
    banned = {'torch.func.hessian', 'torch.autograd.functional.hessian', 'torch.func.jacfwd',
              'func.hessian', 'functional.hessian', 'func.jacfwd'}

    for module in (forces_mod, kinematics, materials_torch, reduced_model, rollout_mod, step):
        tree = ast.parse(inspect.getsource(module))
        called = {ast.unparse(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)}
        offenders = called & banned
        assert not offenders, f'{module.__name__} calls {sorted(offenders)}'
