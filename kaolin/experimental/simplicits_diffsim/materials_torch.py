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

r"""Volume-weighted Neo-Hookean energy, gradient and Hessian in pure PyTorch.

These mirror :class:`kaolin.physics.materials.neohookean_elastic_material.NeohookeanElasticMaterial`
(the Warp path used by ``SimplicitsScene``) term for term, including its
``reparameterize_lame=True`` convention.

Why an analytic Hessian rather than ``torch.func.hessian``: the functorch composition
``jacfwd(jacrev(psi))`` returns *correct Hessian values* but a **wrong gradient when
differentiated through** -- a converged finite-difference check puts the resulting parameter
gradient ~24% off -- and ``vmap(jacrev(jacrev(psi)))`` returns NaN. Since simulation-in-the-loop
training backpropagates through the Hessian, the whole differentiated path is kept first-order:
every function below uses only elementwise ops, ``linalg.inv`` and reshapes, so
``autograd.grad`` on the Hessian is an ordinary first-order backward pass.
"""

import torch

__all__ = [
    'det3x3',
    'neohookean_energy',
    'neohookean_gradient',
    'neohookean_hessian',
]


def det3x3(defo_grad):
    r"""Batched determinant of :math:`3 \times 3` matrices via cofactor expansion.

    Used in place of :func:`torch.det`, whose backward relies on an LU factorization; the explicit
    polynomial keeps higher-order derivatives cheap and exact.

    Args:
        defo_grad (torch.Tensor): Batch of matrices, of shape :math:`(\text{batch_dims}, 3, 3)`.

    Returns:
        torch.Tensor: Determinants, of shape :math:`(\text{batch_dims},)`.
    """
    f = defo_grad
    return (f[..., 0, 0] * (f[..., 1, 1] * f[..., 2, 2] - f[..., 1, 2] * f[..., 2, 1])
            - f[..., 0, 1] * (f[..., 1, 0] * f[..., 2, 2] - f[..., 1, 2] * f[..., 2, 0])
            + f[..., 0, 2] * (f[..., 1, 0] * f[..., 2, 1] - f[..., 1, 1] * f[..., 2, 0]))


def _lame(mu, lam, reparameterize_lame):
    if reparameterize_lame:
        lam = lam + mu
    return mu, lam


def neohookean_energy(defo_grad, mu, lam, vol, reparameterize_lame=True):
    r"""Volume-weighted Neo-Hookean strain energy per integration point,

    .. math::

        \Psi_i = \text{vol}_i \left[ \frac{\mu_i}{2}(I_1 - 3)
                 + \frac{\lambda_i}{2}(J - 1)^2 - \mu_i (J - 1) \right].

    Args:
        defo_grad (torch.Tensor): Deformation gradients, of shape :math:`(\text{num_samples}, 3, 3)`.
        mu (torch.Tensor): Lame :math:`\mu`, of shape :math:`(\text{num_samples},)`.
        lam (torch.Tensor): Lame :math:`\lambda`, of shape :math:`(\text{num_samples},)`.
        vol (torch.Tensor): Integration volumes, of shape :math:`(\text{num_samples},)`.
        reparameterize_lame (bool, optional): If True, use :math:`\lambda \leftarrow \lambda + \mu`,
            matching ``SimplicitsScene``. Default: True.

    Returns:
        torch.Tensor: Per-point energies, of shape :math:`(\text{num_samples},)`.
    """
    mu, lam = _lame(mu, lam, reparameterize_lame)
    first_invariant = (defo_grad * defo_grad).sum(dim=(-2, -1))
    jacobian = det3x3(defo_grad)
    return vol * (0.5 * mu * (first_invariant - 3.0)
                  + 0.5 * lam * (jacobian - 1.0) ** 2
                  - mu * (jacobian - 1.0))


def neohookean_gradient(defo_grad, mu, lam, vol, reparameterize_lame=True):
    r"""First derivative of :func:`neohookean_energy` with respect to the deformation gradient,

    .. math::

        \frac{\partial \Psi}{\partial F} = \text{vol} \left[ \mu F
            + \left( \lambda (J-1) J - \mu J \right) F^{-T} \right].

    Args:
        defo_grad (torch.Tensor): Deformation gradients, of shape :math:`(\text{num_samples}, 3, 3)`.
        mu (torch.Tensor): Lame :math:`\mu`, of shape :math:`(\text{num_samples},)`.
        lam (torch.Tensor): Lame :math:`\lambda`, of shape :math:`(\text{num_samples},)`.
        vol (torch.Tensor): Integration volumes, of shape :math:`(\text{num_samples},)`.
        reparameterize_lame (bool, optional): If True, use :math:`\lambda \leftarrow \lambda + \mu`.
            Default: True.

    Returns:
        torch.Tensor: Per-point gradients, of shape :math:`(\text{num_samples}, 3, 3)`.
    """
    mu, lam = _lame(mu, lam, reparameterize_lame)
    mu_c = mu.reshape(-1, 1, 1)
    lam_c = lam.reshape(-1, 1, 1)
    jacobian = det3x3(defo_grad).reshape(-1, 1, 1)
    inv_transpose = torch.linalg.inv(defo_grad).transpose(-2, -1)
    coeff = lam_c * (jacobian - 1.0) * jacobian - mu_c * jacobian
    return vol.reshape(-1, 1, 1) * (mu_c * defo_grad + coeff * inv_transpose)


def neohookean_hessian(defo_grad, mu, lam, vol, reparameterize_lame=True):
    r"""Second derivative of :func:`neohookean_energy` with respect to the deformation gradient,
    as dense :math:`9 \times 9` blocks in row-major :math:`F` ordering (entry :math:`(i, j)` of
    :math:`F` maps to index :math:`3i + j`).

    Writing :math:`g = F^{-T}` (so :math:`\partial J / \partial F = J g`) and using
    :math:`\partial^2 J / \partial F_{ab} \partial F_{cd} = J (g_{ab} g_{cd} - g_{ad} g_{cb})`,

    .. math::

        \frac{\partial^2 \Psi}{\partial F^2} = \text{vol} \left[ \mu I_9
            + \gamma \, g \otimes g - \tilde{\gamma} \, P(g \otimes g) \right],

    with :math:`\gamma = J(\lambda(2J - 1) - \mu)`, :math:`\tilde{\gamma} = \gamma - \lambda J^2`
    and :math:`P` the index transposition :math:`(a, b, c, d) \mapsto (a, d, c, b)`.

    The blocks are consumed by :func:`kaolin.physics.utils.torch_utilities.hess_reduction` to form
    :math:`\left(\partial F / \partial z\right)^T H \left(\partial F / \partial z\right)`.

    Args:
        defo_grad (torch.Tensor): Deformation gradients, of shape :math:`(\text{num_samples}, 3, 3)`.
        mu (torch.Tensor): Lame :math:`\mu`, of shape :math:`(\text{num_samples},)`.
        lam (torch.Tensor): Lame :math:`\lambda`, of shape :math:`(\text{num_samples},)`.
        vol (torch.Tensor): Integration volumes, of shape :math:`(\text{num_samples},)`.
        reparameterize_lame (bool, optional): If True, use :math:`\lambda \leftarrow \lambda + \mu`.
            Default: True.

    Returns:
        torch.Tensor: Per-point Hessian blocks, of shape :math:`(\text{num_samples}, 9, 9)`.
    """
    mu, lam = _lame(mu, lam, reparameterize_lame)
    num_samples = defo_grad.shape[0]
    mu_c = mu.reshape(num_samples, 1, 1)
    lam_c = lam.reshape(num_samples, 1, 1)

    jacobian = det3x3(defo_grad).reshape(num_samples, 1, 1)
    inv_transpose = torch.linalg.inv(defo_grad).transpose(-2, -1)
    grad_j = inv_transpose.reshape(num_samples, 9)

    gamma = jacobian * (lam_c * (2.0 * jacobian - 1.0) - mu_c)
    gamma_tilde = gamma - lam_c * jacobian * jacobian

    outer = grad_j.unsqueeze(-1) * grad_j.unsqueeze(-2)
    transposed = outer.reshape(num_samples, 3, 3, 3, 3).permute(0, 1, 4, 3, 2)
    transposed = transposed.reshape(num_samples, 9, 9)

    eye9 = torch.eye(9, device=defo_grad.device, dtype=defo_grad.dtype)
    return vol.reshape(num_samples, 1, 1) * (mu_c * eye9 + gamma * outer - gamma_tilde * transposed)
