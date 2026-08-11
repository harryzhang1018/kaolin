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

r"""Snapshot projection: the best a given basis could possibly do.

Given full-order displacements :math:`u`, the mass-weighted least-squares fit

.. math::

    z^*(\theta) = \arg\min_z \| u - B_\theta z \|_M^2 + \varepsilon \|z\|^2

has a closed form, so its value *and* its gradient with respect to :math:`\theta` are cheap. Two
uses:

* **A diagnostic floor.** Rollout error can never beat projection error. If a trained field's
  rollout error is close to its projection error, the dynamics are being tracked as well as the
  basis allows and further gains require more handles, not better training.
* **A pretraining objective and baseline.** Fitting :math:`B_\theta` to snapshots directly (the
  "variable projection" objective) is much cheaper than rolling out, and is the natural comparison
  point against POD, which optimizes exactly this quantity over unconstrained linear bases.
"""

import torch

__all__ = [
    'best_fit_reduced_coords',
    'projection_error',
    'pod_basis',
]


def best_fit_reduced_coords(lbs, displacements, sample_masses=None, ridge=0.0):
    r"""Mass-weighted least-squares fit of reduced coordinates to full-order displacements.

    Solves the normal equations :math:`(B^T M B + \varepsilon I) z = B^T M u`, which is
    differentiable in :math:`B` (hence in :math:`\theta`) through
    :func:`torch.linalg.solve`.

    Args:
        lbs (torch.Tensor): Linear-blend-skinning matrix evaluated at the full-order nodes,
            of shape :math:`(3 \text{num_pts}, 12 \text{num_handles})`.
        displacements (torch.Tensor): Full-order displacements :math:`x - X`, of shape
            :math:`(\text{num_frames}, \text{num_pts}, 3)` or :math:`(\text{num_pts}, 3)`.
        sample_masses (torch.Tensor, optional): Per-node masses or nodal volumes, of shape
            :math:`(\text{num_pts},)`. Default: None (unweighted).
        ridge (float, optional): Tikhonov regularization :math:`\varepsilon`. Default: 0.0.

    Returns:
        torch.Tensor: Reduced coordinates, of shape
        :math:`(\text{num_frames}, 12 \text{num_handles})` or
        :math:`(12 \text{num_handles},)`, matching the input rank.
    """
    single = displacements.dim() == 2
    flat = displacements.reshape(1, -1) if single else displacements.reshape(displacements.shape[0], -1)

    if sample_masses is None:
        weighted_lbs = lbs
    else:
        weighted_lbs = sample_masses.repeat_interleave(3).unsqueeze(1) * lbs

    normal = lbs.transpose(0, 1) @ weighted_lbs
    if ridge != 0.0:
        eye = torch.eye(normal.shape[0], device=normal.device, dtype=normal.dtype)
        normal = normal + ridge * eye

    rhs = weighted_lbs.transpose(0, 1) @ flat.transpose(0, 1)
    coords = torch.linalg.solve(normal, rhs).transpose(0, 1)
    return coords[0] if single else coords


def projection_error(lbs, displacements, rest_pts=None, sample_masses=None, ridge=0.0):
    r"""Best-projection error of a basis onto a set of full-order snapshots.

    This is the error floor for any rollout in the same basis; reporting it alongside rollout
    error separates "the basis cannot represent this motion" from "training failed to track it".

    Args:
        lbs (torch.Tensor): Linear-blend-skinning matrix evaluated at the full-order nodes,
            of shape :math:`(3 \text{num_pts}, 12 \text{num_handles})`.
        displacements (torch.Tensor): Full-order displacements, of shape
            :math:`(\text{num_frames}, \text{num_pts}, 3)`.
        rest_pts (torch.Tensor, optional): Rest positions, of shape :math:`(\text{num_pts}, 3)`.
            Only needed if you want the reconstructed positions back. Default: None.
        sample_masses (torch.Tensor, optional): Per-node masses. Default: None.
        ridge (float, optional): Tikhonov regularization. Default: 0.0.

    Returns:
        (torch.Tensor, torch.Tensor, torch.Tensor): the mean squared projection error (scalar),
        the fitted coordinates of shape :math:`(\text{num_frames}, 12 \text{num_handles})`, and
        the reconstruction -- displacements, or positions if ``rest_pts`` was given -- of shape
        :math:`(\text{num_frames}, \text{num_pts}, 3)`.
    """
    coords = best_fit_reduced_coords(lbs, displacements, sample_masses, ridge)
    recon = (lbs @ coords.transpose(0, 1)).transpose(0, 1).reshape(displacements.shape)
    error = ((recon - displacements) ** 2).sum(-1).mean()
    if rest_pts is not None:
        recon = rest_pts + recon
    return error, coords, recon


def pod_basis(displacements, num_modes, sample_masses=None):
    r"""Proper-orthogonal-decomposition basis of a snapshot matrix, as an unconstrained-linear-basis
    baseline.

    A rank-:math:`r` POD basis is the optimal :math:`r`-dimensional linear subspace for these
    snapshots in the given norm, so it upper-bounds what any skinning field with the same number of
    reduced coordinates can achieve at *representation*. Simplicits trades some of that
    representation power for a basis that is continuous in space, defined off the snapshots, and
    only :math:`12H` wide.

    Args:
        displacements (torch.Tensor): Full-order displacements, of shape
            :math:`(\text{num_frames}, \text{num_pts}, 3)`.
        num_modes (int): Number of modes :math:`r` to keep.
        sample_masses (torch.Tensor, optional): Per-node masses defining the norm.
            Default: None (Euclidean).

    Returns:
        (torch.Tensor, torch.Tensor): the basis of shape :math:`(3 \text{num_pts}, r)` and the
        full singular-value spectrum of shape :math:`(\min(\text{num_frames}, 3\text{num_pts}),)`.
    """
    snapshots = displacements.reshape(displacements.shape[0], -1).transpose(0, 1)
    if sample_masses is None:
        scale = None
        scaled = snapshots
    else:
        scale = sample_masses.repeat_interleave(3).sqrt().unsqueeze(1)
        scaled = scale * snapshots

    left, singular_values, _ = torch.linalg.svd(scaled, full_matrices=False)
    basis = left[:, :num_modes]
    if scale is not None:
        basis = basis / scale
    return basis, singular_values
