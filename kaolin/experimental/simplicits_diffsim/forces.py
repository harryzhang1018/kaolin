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

r"""Point-wise external potentials in pure PyTorch.

Each force mirrors the corresponding Warp struct in
:mod:`kaolin.physics.common.scene_forces` -- including how ``SimplicitsScene`` parameterizes it,
which is not always obvious from the kernel alone:

* ``Gravity`` is built with ``integration_pt_volume=sim_vols`` and coefficient 1.
* ``Floor`` is built with ``integration_pt_volume=ones_like(sim_vols)``, so the penalty is
  *not* volume-weighted, and its coefficient is ``floor_penalty``.
* ``Boundary`` ignores volume entirely in the kernel; its coefficient is ``bdry_penalty``.

All three act on positions :math:`x = X + Bz`, so they enter the reduced system through
:math:`B^T`. Each exposes ``energy``, ``gradient`` (per-point :math:`3`-vectors) and
``hessian_blocks`` (per-point :math:`3 \times 3`).
"""

import torch

__all__ = [
    'Gravity',
    'Floor',
    'Boundary',
]


class Gravity:
    r"""Gravitational potential :math:`E = \sum_i \rho_i \text{vol}_i \, g \cdot x_i`.

    Note the sign convention of ``SimplicitsScene``: the scene is set up with a *positive*
    ``[0, 9.8, 0]`` acceleration vector and the energy is :math:`+ m g \cdot x`, so the resulting
    force pushes along :math:`-g`, i.e. downward in :math:`-y`.

    Args:
        gravity (torch.Tensor): Acceleration vector, of shape :math:`(3,)` (in :math:`m/s^2`).
        rhos (torch.Tensor): Per-point densities, of shape :math:`(\text{num_samples},)`
            (in :math:`kg/m^3`).
        vols (torch.Tensor): Per-point integration volumes, of shape :math:`(\text{num_samples},)`
            (in :math:`m^3`).
        coeff (float, optional): Scaling coefficient. Default: 1.0.
    """

    def __init__(self, gravity, rhos, vols, coeff=1.0):
        self.gravity = gravity
        self.point_masses = rhos * vols
        self.coeff = coeff

    def energy(self, positions):
        r"""Total gravitational energy.

        Args:
            positions (torch.Tensor): Deformed positions, of shape :math:`(\text{num_samples}, 3)`.

        Returns:
            torch.Tensor: Scalar energy.
        """
        return self.coeff * (self.point_masses * (positions * self.gravity).sum(-1)).sum()

    def gradient(self, positions):
        r"""Per-point gradient of :meth:`energy`.

        Args:
            positions (torch.Tensor): Deformed positions, of shape :math:`(\text{num_samples}, 3)`.

        Returns:
            torch.Tensor: Gradient, of shape :math:`(\text{num_samples}, 3)`.
        """
        return self.coeff * self.point_masses.unsqueeze(-1) * self.gravity

    def hessian_blocks(self, positions):
        r"""Per-point Hessian blocks of :meth:`energy`, identically zero (gravity is linear).

        Args:
            positions (torch.Tensor): Deformed positions, of shape :math:`(\text{num_samples}, 3)`.

        Returns:
            torch.Tensor: Zeros, of shape :math:`(\text{num_samples}, 3, 3)`.
        """
        return positions.new_zeros(positions.shape[0], 3, 3)


class Floor:
    r"""One-sided quadratic floor penalty
    :math:`E = k \sum_i \left[\max(0, h - x_{i,a})\right]^2` along axis :math:`a`
    (or :math:`\max(0, x_{i,a} - h)` when flipped).

    Non-smooth at activation: the Hessian jumps between :math:`0` and :math:`2k`. Training should
    keep the object clear of the floor until the smooth case is verified.

    Args:
        floor_height (float): Floor position along ``floor_axis`` (in :math:`m`).
        floor_axis (int, optional): Axis index, 0 for x, 1 for y, 2 for z. Default: 1.
        flip_floor (bool, optional): If True, penalize positions *above* the floor.
            Default: False.
        coeff (float, optional): Penalty stiffness, matching ``floor_penalty``. Default: 10000.0.
    """

    def __init__(self, floor_height, floor_axis=1, flip_floor=False, coeff=10000.0):
        self.floor_height = floor_height
        self.floor_axis = floor_axis
        self.flip_floor = flip_floor
        self.coeff = coeff

    def _violation(self, positions):
        along = positions[:, self.floor_axis] - self.floor_height
        return -along if self.flip_floor else along

    def energy(self, positions):
        r"""Total floor penalty energy.

        Args:
            positions (torch.Tensor): Deformed positions, of shape :math:`(\text{num_samples}, 3)`.

        Returns:
            torch.Tensor: Scalar energy.
        """
        return self.coeff * (torch.clamp(-self._violation(positions), min=0.0) ** 2).sum()

    def gradient(self, positions):
        r"""Per-point gradient of :meth:`energy`.

        Args:
            positions (torch.Tensor): Deformed positions, of shape :math:`(\text{num_samples}, 3)`.

        Returns:
            torch.Tensor: Gradient, of shape :math:`(\text{num_samples}, 3)`.
        """
        along = self._violation(positions)
        active = (along < 0.0).to(positions.dtype)
        out = positions.new_zeros(positions.shape)
        sign = -1.0 if self.flip_floor else 1.0
        out[:, self.floor_axis] = sign * 2.0 * self.coeff * along * active
        return out

    def hessian_blocks(self, positions):
        r"""Per-point Hessian blocks of :meth:`energy`.

        Args:
            positions (torch.Tensor): Deformed positions, of shape :math:`(\text{num_samples}, 3)`.

        Returns:
            torch.Tensor: Blocks, of shape :math:`(\text{num_samples}, 3, 3)`.
        """
        active = (self._violation(positions) < 0.0).to(positions.dtype)
        out = positions.new_zeros(positions.shape[0], 3, 3)
        out[:, self.floor_axis, self.floor_axis] = 2.0 * self.coeff * active
        return out


class Boundary:
    r"""Quadratic pin penalty :math:`E = k \sum_{i \in \mathcal{P}} \|x_i - p_i\|^2` on a subset of
    points.

    Args:
        pinned_indices (torch.Tensor): Indices of pinned sample points, of shape
            :math:`(\text{num_pinned},)`.
        pinned_positions (torch.Tensor): Target positions, of shape
            :math:`(\text{num_pinned}, 3)`.
        coeff (float, optional): Penalty stiffness, matching ``bdry_penalty``. Default: 10000.0.
    """

    def __init__(self, pinned_indices, pinned_positions, coeff=10000.0):
        self.pinned_indices = pinned_indices
        self.pinned_positions = pinned_positions
        self.coeff = coeff

    def energy(self, positions):
        r"""Total pin penalty energy.

        Args:
            positions (torch.Tensor): Deformed positions, of shape :math:`(\text{num_samples}, 3)`.

        Returns:
            torch.Tensor: Scalar energy.
        """
        delta = positions[self.pinned_indices] - self.pinned_positions
        return self.coeff * (delta * delta).sum()

    def gradient(self, positions):
        r"""Per-point gradient of :meth:`energy`.

        Args:
            positions (torch.Tensor): Deformed positions, of shape :math:`(\text{num_samples}, 3)`.

        Returns:
            torch.Tensor: Gradient, of shape :math:`(\text{num_samples}, 3)`.
        """
        delta = positions[self.pinned_indices] - self.pinned_positions
        out = positions.new_zeros(positions.shape)
        return out.index_add(0, self.pinned_indices, 2.0 * self.coeff * delta)

    def hessian_blocks(self, positions):
        r"""Per-point Hessian blocks of :meth:`energy`.

        Args:
            positions (torch.Tensor): Deformed positions, of shape :math:`(\text{num_samples}, 3)`.

        Returns:
            torch.Tensor: Blocks, of shape :math:`(\text{num_samples}, 3, 3)`.
        """
        eye3 = torch.eye(3, device=positions.device, dtype=positions.dtype)
        out = positions.new_zeros(positions.shape[0], 3, 3)
        return out.index_add(0, self.pinned_indices,
                             (2.0 * self.coeff * eye3).expand(self.pinned_indices.shape[0], 3, 3))
