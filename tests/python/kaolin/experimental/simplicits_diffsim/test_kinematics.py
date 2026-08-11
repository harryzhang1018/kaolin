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

"""Layer 1 of the equivalence ladder: the reduced-order kinematic operators."""

import pytest
import torch
import warp as wp

from kaolin.physics.simplicits.network import SimplicitsMLP
from kaolin.physics.simplicits.precomputed import sparse_dFdz_matrix
from kaolin.physics.utils import _wp_bsr_to_torch_bsr
from kaolin.experimental.simplicits_diffsim.kinematics import (dense_dFdz_matrix, dense_lbs_matrix,
                                                              qr_reparameterization,
                                                              reduced_mass_matrix)


@pytest.mark.parametrize('device', ['cuda'])
def test_dense_dFdz_matches_warp_sparse(device):
    """The torch dF/dz must reproduce the Warp triplet kernel entry for entry, including layout."""
    torch.manual_seed(0)
    num_samples, num_handles = 64, 4
    skinning_mod = SimplicitsMLP(3, 8, num_handles, 1, bb_min=torch.zeros(3),
                                 bb_max=torch.ones(3)).to(device)
    pts = torch.rand(num_samples, 3, device=device)
    weights = skinning_mod.compute_skinning_weights(pts)
    weights_jac = skinning_mod.compute_dwdx(pts)

    expected = _wp_bsr_to_torch_bsr(sparse_dFdz_matrix(
        wp.from_torch(weights.detach().contiguous()),
        wp.from_torch(weights_jac.detach().contiguous()),
        wp.from_torch(pts.contiguous(), dtype=wp.vec3))).to_dense()
    actual = dense_dFdz_matrix(pts, weights, weights_jac)

    assert actual.shape == (9 * num_samples, 12 * num_handles)
    assert (actual - expected).abs().max() / expected.abs().max() < 1e-6


@pytest.mark.parametrize('device', ['cuda'])
def test_dense_dFdz_backprops_to_parameters(device):
    """dF/dz must stay attached to theta -- the production ``jacobian_dF_dz`` does not."""
    torch.manual_seed(0)
    skinning_mod = SimplicitsMLP(3, 8, 4, 1, bb_min=torch.zeros(3), bb_max=torch.ones(3)).to(device)
    pts = torch.rand(32, 3, device=device)
    dfdz = dense_dFdz_matrix(pts, skinning_mod.compute_skinning_weights(pts),
                             skinning_mod.compute_dwdx(pts))
    grads = torch.autograd.grad(dfdz.sum(), list(skinning_mod.parameters()), allow_unused=True)
    assert all(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads)


@pytest.mark.parametrize('device', ['cuda'])
def test_operators_match_baked_scene(device, make_matched_pair):
    """B, dF/dz and B^T M B must match what the scene baked for its own simulation."""
    pair = make_matched_pair(device)
    sim_object = pair.sim_object

    assert (pair.model.lbs - sim_object.B_dense).abs().max() / sim_object.B_dense.abs().max() < 1e-6
    warp_dfdz = sim_object.dFdz_dense
    assert (pair.model.dFdz - warp_dfdz).abs().max() / warp_dfdz.abs().max() < 1e-6

    warp_bmb = _wp_bsr_to_torch_bsr(pair.scene.sim_BMB).to_dense()
    assert (pair.model.reduced_mass - warp_bmb).abs().max() / warp_bmb.abs().max() < 1e-6


@pytest.mark.parametrize('device', ['cuda'])
def test_dense_dFdz_is_lbs_jacobian_in_float64(device):
    """dF/dz must equal d(Bz)/dX, i.e. the spatial Jacobian of the skinned positions."""
    torch.manual_seed(0)
    num_samples, num_handles = 16, 3
    skinning_mod = SimplicitsMLP(3, 8, num_handles, 1, bb_min=torch.zeros(3),
                                 bb_max=torch.ones(3)).to(device=device, dtype=torch.float64)
    pts = torch.rand(num_samples, 3, device=device, dtype=torch.float64)
    coords = (torch.rand(12 * num_handles, device=device, dtype=torch.float64) - 0.5) * 0.1

    dfdz = dense_dFdz_matrix(pts, skinning_mod.compute_skinning_weights(pts),
                             skinning_mod.compute_dwdx(pts))
    expected = (dfdz @ coords).reshape(num_samples, 3, 3)

    def deformed(single_pt):
        pt = single_pt[None]
        lbs = dense_lbs_matrix(pt, skinning_mod.compute_skinning_weights(pt))
        return (lbs @ coords).reshape(3)

    actual = torch.vmap(torch.func.jacrev(deformed))(pts)
    assert (actual - expected).abs().max() < 1e-9


@pytest.mark.parametrize('device', ['cuda'])
def test_reduced_mass_and_qr(device):
    """B^T M B is the lumped-mass reduction; the QR factor orthonormalizes B's columns."""
    torch.manual_seed(0)
    num_samples, num_handles = 32, 3
    pts = torch.rand(num_samples, 3, device=device, dtype=torch.float64)
    weights = torch.rand(num_samples, num_handles, device=device, dtype=torch.float64)
    lbs = dense_lbs_matrix(pts, weights)
    masses = torch.rand(num_samples, device=device, dtype=torch.float64) + 0.5

    expected = lbs.transpose(0, 1) @ torch.diag(masses.repeat_interleave(3)) @ lbs
    assert (reduced_mass_matrix(lbs, masses) - expected).abs().max() < 1e-12

    qr_tfm = qr_reparameterization(lbs)
    orthonormal = lbs @ qr_tfm
    gram = orthonormal.transpose(0, 1) @ orthonormal
    eye = torch.eye(gram.shape[0], device=device, dtype=torch.float64)
    assert (gram - eye).abs().max() < 1e-10
    assert not qr_tfm.requires_grad
