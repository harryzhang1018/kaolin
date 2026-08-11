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

r"""Differentiable reduced-order kinematics.

Every quantity here is a pure-PyTorch function of the skinning weights
:math:`W_\theta(X)` and their spatial Jacobian :math:`\nabla_X W_\theta(X)`, so gradients flow
back to the network parameters :math:`\theta`. The production Warp path
(:func:`kaolin.physics.simplicits.precomputed.sparse_dFdz_matrix` and friends) builds the same
matrices but severs the autograd graph at the ``wp.from_torch`` boundary.
"""

import torch

from kaolin.physics.simplicits.precomputed import lbs_matrix

__all__ = [
    'dense_lbs_matrix',
    'dense_dFdz_matrix',
    'reduced_mass_matrix',
    'qr_reparameterization',
    'uniform_sample_volumes',
]


def dense_lbs_matrix(pts, weights):
    r"""Dense, differentiable linear-blend-skinning matrix :math:`B`, such that
    :math:`\text{flatten}(dx) = B z` where :math:`z` stacks the :math:`3 \times 4` affine handle
    transforms.

    This is a thin alias of :func:`kaolin.physics.simplicits.precomputed.lbs_matrix`, which is
    already pure PyTorch and differentiable. It exists so that callers of this module never need
    to reach into ``kaolin.physics.simplicits.precomputed`` for one half of the pair.

    Args:
        pts (torch.Tensor): Rest positions, of shape :math:`(\text{num_samples}, 3)`.
        weights (torch.Tensor): Skinning weights (including the constant handle),
            of shape :math:`(\text{num_samples}, \text{num_handles})`.

    Returns:
        torch.Tensor: Matrix of shape :math:`(3 \text{num_samples}, 12 \text{num_handles})`.
    """
    return lbs_matrix(pts, weights)


def dense_dFdz_matrix(pts, weights, weights_jac):
    r"""Dense, differentiable Jacobian of the deformation gradient with respect to the reduced
    coordinates, :math:`\partial F / \partial z`.

    For sample point :math:`i`, handle :math:`h`, diagonal block :math:`b` (a row of
    :math:`T_h`) and entry :math:`(m, n)` of that :math:`3 \times 4` sub-block, the closed form is

    .. math::

        \left[\frac{\partial F}{\partial z}\right] = \begin{cases}
            w_h + x_m \, \partial_m w_h & m = n \\
            \partial_m w_h & n = 3 \\
            x_n \, \partial_m w_h & \text{otherwise}
        \end{cases}

    which is exactly what the Warp kernel ``_get_dFdz_triplets_wp_kernel``
    (``kaolin/physics/simplicits/precomputed.py``) writes into its triplets. Rows are ordered
    :math:`9i + 3b + m` and columns :math:`12h + 4b + n`, matching that kernel, so
    ``(dFdz @ z).reshape(-1, 3, 3)`` is the row-major per-point deformation-gradient delta.

    Unlike :func:`kaolin.physics.simplicits.precomputed.jacobian_dF_dz`, which calls
    ``torch.autograd.functional.jacobian`` without ``create_graph=True`` and therefore detaches
    :math:`\theta`, this construction is a plain tensor expression and backpropagates.

    Args:
        pts (torch.Tensor): Rest positions, of shape :math:`(\text{num_samples}, 3)`.
        weights (torch.Tensor): Skinning weights (including the constant handle),
            of shape :math:`(\text{num_samples}, \text{num_handles})`.
        weights_jac (torch.Tensor): Spatial gradient of the skinning weights,
            of shape :math:`(\text{num_samples}, \text{num_handles}, 3)`.

    Returns:
        torch.Tensor: Matrix of shape :math:`(9 \text{num_samples}, 12 \text{num_handles})`.
    """
    if weights.shape != weights_jac.shape[:2] or weights_jac.shape[2] != 3:
        raise ValueError(f'Shape mismatch: weights {tuple(weights.shape)} vs '
                         f'weights_jac {tuple(weights_jac.shape)}')
    num_samples, num_handles = weights.shape
    eye3 = torch.eye(3, device=pts.device, dtype=pts.dtype)

    # (N, H, 3, 3), entry [., h, m, n] = x_n * dw_h/dx_m + w_h * delta_mn
    linear = pts.unsqueeze(1).unsqueeze(1) * weights_jac.unsqueeze(-1)
    linear = linear + weights.unsqueeze(-1).unsqueeze(-1) * eye3
    # (N, H, 3, 4), appending the translation column dw_h/dx_m
    sub_block = torch.cat([linear, weights_jac.unsqueeze(-1)], dim=-1)
    # Replicate the 3x4 sub-block onto the three diagonal blocks: rows 3b+m, columns 4b+n.
    blocks = torch.einsum('bc,nhmk->nbmhck', eye3, sub_block)
    return blocks.reshape(9 * num_samples, 12 * num_handles)


def reduced_mass_matrix(lbs, sample_masses):
    r"""Reduced mass matrix :math:`B^T M B` for a lumped (diagonal) point mass matrix.

    Args:
        lbs (torch.Tensor): Linear-blend-skinning matrix :math:`B`,
            of shape :math:`(3 \text{num_samples}, 12 \text{num_handles})`.
        sample_masses (torch.Tensor): Per-point masses, of shape :math:`(\text{num_samples},)`
            (in :math:`kg`).

    Returns:
        torch.Tensor: Matrix of shape :math:`(12 \text{num_handles}, 12 \text{num_handles})`.
    """
    mass_diag = sample_masses.repeat_interleave(3).unsqueeze(1)
    return lbs.transpose(0, 1) @ (mass_diag * lbs)


def qr_reparameterization(lbs):
    r"""Right factor :math:`K = R^{-1}` of a thin QR decomposition :math:`B = QR`, so that
    :math:`BK` has orthonormal columns and :math:`z = K z'` reparameterizes the reduced
    coordinates for conditioning.

    This mirrors ``SimulatedObject._apply_qr_decomposition``. The factor is always computed
    without a gradient, matching the production path (which computes it once at bake time from
    detached weights) and keeping the reparameterization a fixed linear change of variables
    rather than a :math:`\theta`-dependent one.

    Args:
        lbs (torch.Tensor): Linear-blend-skinning matrix :math:`B`,
            of shape :math:`(3 \text{num_samples}, 12 \text{num_handles})`.

    Returns:
        torch.Tensor: Matrix of shape :math:`(12 \text{num_handles}, 12 \text{num_handles})`.
    """
    with torch.no_grad():
        _, upper = torch.linalg.qr(lbs)
        eye = torch.eye(upper.shape[0], device=lbs.device, dtype=lbs.dtype)
        return torch.linalg.solve_triangular(upper, eye, upper=True)


def uniform_sample_volumes(num_samples, appx_vol, device=None, dtype=torch.float32):
    r"""Per-point integration volumes for a spatially uniform Monte-Carlo cubature,
    :math:`\text{vol}_i = V / N`, matching ``SimplicitsScene``'s ``sim_vols``.

    Args:
        num_samples (int): Number of quadrature points :math:`N`.
        appx_vol (float or torch.Tensor): Total object volume :math:`V` (in :math:`m^3`).
        device (torch.device, optional): Output device. Default: None.
        dtype (torch.dtype, optional): Output dtype. Default: ``torch.float32``.

    Returns:
        torch.Tensor: Per-point volumes, of shape :math:`(\text{num_samples},)`.
    """
    total = float(appx_vol)
    return torch.full((num_samples,), total / num_samples, device=device, dtype=dtype)
