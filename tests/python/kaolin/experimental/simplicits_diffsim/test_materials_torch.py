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

"""Layer 2 of the equivalence ladder: the Neo-Hookean constitutive model.

The analytic Hessian is checked three ways -- against the Warp kernel, against autograd of the
analytic gradient, and (third derivative) against finite differences of a Hessian contraction.
The last one is what actually matters for simulation-in-the-loop training, because Stage A
backpropagates *through* the Hessian.
"""

import pytest
import torch
import warp as wp

from kaolin.physics.materials.neohookean_elastic_material import (NeohookeanElasticMaterial,
                                                                 _neohookean_energy,
                                                                 _neohookean_gradient)
from kaolin.experimental.simplicits_diffsim.materials_torch import (det3x3, neohookean_energy,
                                                                   neohookean_gradient,
                                                                   neohookean_hessian)

NUM_SAMPLES = 128
YOUNGS = 1e5
POISSON = 0.45


def _random_defo_grads(device, dtype, num_samples=NUM_SAMPLES, scale=0.15, seed=0):
    """Deformation gradients near identity, so J stays safely positive."""
    torch.manual_seed(seed)
    return torch.eye(3, device=device, dtype=dtype) + scale * torch.randn(
        num_samples, 3, 3, device=device, dtype=dtype)


def _lame(device, dtype, num_samples=NUM_SAMPLES):
    from kaolin.physics.materials.material_utils import to_lame
    yms = torch.full((num_samples,), YOUNGS, device=device, dtype=dtype)
    prs = torch.full((num_samples,), POISSON, device=device, dtype=dtype)
    mus, lams = to_lame(yms, prs)
    vols = torch.full((num_samples,), 1.0 / num_samples, device=device, dtype=dtype)
    return mus, lams, vols


@pytest.mark.parametrize('device', ['cuda'])
def test_det3x3_matches_torch_det(device):
    """The cofactor-expansion determinant must agree with ``torch.det``."""
    defo_grads = _random_defo_grads(device, torch.float64)
    assert (det3x3(defo_grads) - torch.det(defo_grads)).abs().max() < 1e-12


@pytest.mark.parametrize('device', ['cuda'])
def test_energy_gradient_hessian_match_warp(device):
    """All three derivatives must match the Warp kernels the scene actually runs."""
    dtype = torch.float32
    defo_grads = _random_defo_grads(device, dtype)
    mus, lams, vols = _lame(device, dtype)

    material = NeohookeanElasticMaterial(mu=wp.from_torch(mus.contiguous()),
                                         lam=wp.from_torch(lams.contiguous()),
                                         integration_pt_volume=wp.from_torch(vols.contiguous()),
                                         reparameterize_lame=True)
    wp_defo_grads = wp.from_torch(defo_grads.contiguous(), dtype=wp.mat33)

    warp_energy = float(wp.to_torch(material.energy(wp_defo_grads)).sum())
    torch_energy = float(neohookean_energy(defo_grads, mus, lams, vols).sum())
    assert abs(torch_energy - warp_energy) / abs(warp_energy) < 1e-5

    # NOTE: use the default output buffer. ``NeohookeanElasticMaterial.gradients`` is preallocated
    # as vec9 while ``_neohookean_gradient_wp_kernel`` declares its output as mat33, so passing the
    # preallocated buffer raises. Upstream quirk; left alone deliberately.
    warp_grad = wp.to_torch(material.gradient(wp_defo_grads)).clone().reshape(-1, 3, 3)
    torch_grad = neohookean_gradient(defo_grads, mus, lams, vols)
    assert (torch_grad - warp_grad).abs().max() / warp_grad.abs().max() < 1e-5

    warp_hess = wp.to_torch(material.hessian(wp_defo_grads)).clone().reshape(-1, 9, 9)
    torch_hess = neohookean_hessian(defo_grads, mus, lams, vols)
    assert (torch_hess - warp_hess).abs().max() / warp_hess.abs().max() < 1e-5


@pytest.mark.parametrize('device', ['cuda'])
def test_energy_matches_upstream_torch_reference(device):
    """The energy must match the upstream torch reference for both Lame conventions."""
    dtype = torch.float64
    defo_grads = _random_defo_grads(device, dtype)
    mus, lams, vols = _lame(device, dtype)
    unit_vols = torch.ones_like(vols)

    for reparameterize in (False, True):
        expected = _neohookean_energy(mus.unsqueeze(-1), lams.unsqueeze(-1), defo_grads,
                                      reparameterize_lame=reparameterize).squeeze(-1)
        actual = neohookean_energy(defo_grads, mus, lams, unit_vols,
                                   reparameterize_lame=reparameterize)
        assert (actual - expected).abs().max() / expected.abs().max() < 1e-12


@pytest.mark.parametrize('device', ['cuda'])
def test_gradient_disagrees_with_buggy_upstream_torch_reference(device):
    """Pins down a latent bug in the upstream *torch* gradient, so the difference is deliberate.

    ``neohookean_elastic_material._neohookean_gradient`` builds its volumetric term with
    ``torch.linalg.inv(F)`` where :math:`\\partial J/\\partial F = J F^{-T}` requires the
    transpose. Away from :math:`F = I` it therefore disagrees with the gradient of its own energy;
    replacing ``inv(F)`` with ``inv(F).transpose(-2, -1)`` makes it agree to 3e-11.

    The bug is latent: no production path calls it (``loss_elastic`` uses only
    ``_neohookean_energy``, and the simulator uses the Warp kernel, which is correct), and
    ``tests/.../test_neohookean_elastic_material.py::test_neohookean_gradient`` misses it because
    its fixture leaves ``F`` exactly equal to the identity -- the ``+ eps * torch.rand(...)`` line
    is a separate statement, not a continuation -- and :math:`I^{-1} = I^{-T}`.

    This test asserts what is true today. If upstream fixes the transpose, it will fail and should
    be replaced by a plain equality check.
    """
    dtype = torch.float64
    defo_grads = _random_defo_grads(device, dtype, num_samples=16)
    mus, lams, vols = _lame(device, dtype, num_samples=16)
    unit_vols = torch.ones_like(vols)

    leaf = defo_grads.clone().requires_grad_(True)
    truth, = torch.autograd.grad(_neohookean_energy(mus.unsqueeze(-1), lams.unsqueeze(-1),
                                                    leaf).sum(), leaf)
    ours = neohookean_gradient(defo_grads, mus, lams, unit_vols, reparameterize_lame=False)
    upstream = _neohookean_gradient(mus.unsqueeze(-1), lams.unsqueeze(-1), defo_grads)

    assert (ours - truth).abs().max() / truth.abs().max() < 1e-12
    assert (upstream - truth).abs().max() / truth.abs().max() > 1e-3

    inv_transpose = torch.linalg.inv(defo_grads).transpose(-2, -1)
    jacobian = torch.det(defo_grads).reshape(-1, 1, 1)
    mu_c, lam_c = mus.reshape(-1, 1, 1), lams.reshape(-1, 1, 1)
    repaired = mu_c * defo_grads + (lam_c * (jacobian - 1.0) * jacobian
                                    - mu_c * jacobian) * inv_transpose
    assert (repaired - truth).abs().max() / truth.abs().max() < 1e-12


@pytest.mark.parametrize('device', ['cuda'])
def test_gradient_and_hessian_are_derivatives_of_energy(device):
    """The analytic gradient and Hessian must be autograd-consistent with the energy."""
    dtype = torch.float64
    defo_grads = _random_defo_grads(device, dtype, num_samples=16).requires_grad_(True)
    mus, lams, vols = _lame(device, dtype, num_samples=16)

    energy = neohookean_energy(defo_grads, mus, lams, vols).sum()
    autograd_grad, = torch.autograd.grad(energy, defo_grads)
    assert (autograd_grad - neohookean_gradient(defo_grads, mus, lams, vols)).abs().max() < 1e-8

    analytic_hess = neohookean_hessian(defo_grads, mus, lams, vols)
    grad_fn = torch.func.jacrev(lambda f: neohookean_gradient(f[None], mus[:1], lams[:1], vols[:1])
                                .reshape(9))
    autograd_hess = torch.vmap(grad_fn)(defo_grads.detach()).reshape(-1, 9, 9)
    scale = analytic_hess.abs().max()
    assert (analytic_hess - autograd_hess).abs().max() / scale < 1e-9


@pytest.mark.parametrize('device', ['cuda'])
def test_hessian_is_differentiable(device):
    """Backpropagating *through* the Hessian must give the true third derivative.

    This is the check that rules out ``torch.func.hessian``: its values are right but its
    derivative is not, which would silently corrupt every Stage A gradient. The reference is a
    central difference of the same contraction -- computed without ``torch.no_grad()``, since that
    context manager changes what some functorch transforms return.
    """
    dtype = torch.float64
    num_samples = 8
    mus, lams, vols = _lame(device, dtype, num_samples=num_samples)
    torch.manual_seed(3)
    base = _random_defo_grads(device, dtype, num_samples=num_samples, scale=0.1)
    probe = torch.randn(num_samples, 9, device=device, dtype=dtype)
    direction = torch.randn_like(base)
    direction = direction / direction.norm()

    def contraction(defo_grads):
        hess = neohookean_hessian(defo_grads, mus, lams, vols)
        return torch.einsum('nab,na,nb->', hess, probe, probe)

    leaf = base.clone().requires_grad_(True)
    analytic, = torch.autograd.grad(contraction(leaf), leaf)
    analytic_directional = float((analytic * direction).sum())

    best = float('inf')
    for eps in (1e-4, 1e-5, 1e-6):
        plus = float(contraction(base + eps * direction))
        minus = float(contraction(base - eps * direction))
        fd = (plus - minus) / (2.0 * eps)
        best = min(best, abs(fd - analytic_directional) / abs(analytic_directional))
    assert best < 1e-7, f'best relative error over the eps sweep was {best:.3e}'


@pytest.mark.parametrize('device', ['cuda'])
def test_hessian_is_symmetric(device):
    """Per-point Hessian blocks must be symmetric."""
    defo_grads = _random_defo_grads(device, torch.float64)
    mus, lams, vols = _lame(device, torch.float64)
    hess = neohookean_hessian(defo_grads, mus, lams, vols)
    assert (hess - hess.transpose(-2, -1)).abs().max() / hess.abs().max() < 1e-12
