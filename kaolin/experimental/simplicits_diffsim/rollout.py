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

r"""Differentiable multi-step rollout with truncated backpropagation through time.

The reduced dynamics are integrated at the *quadrature* points held by a
:class:`~.reduced_model.ReducedModel`, but the trajectory loss is measured at the *full-order
mesh nodes*. :class:`SkinningDecoder` bridges the two by evaluating the same skinning field at the
mesh rest positions, so a gradient reaches :math:`\theta` through both the dynamics and the
decode.
"""

import torch

from .kinematics import dense_lbs_matrix
from .step import newton_step_unrolled

__all__ = [
    'SkinningDecoder',
    'rollout',
]


class SkinningDecoder:
    r"""Maps reduced coordinates to positions at an arbitrary set of rest points.

    Used to evaluate the reduced trajectory at full-order mesh nodes, which are generally not the
    quadrature points that drive the dynamics.

    Args:
        skinning_mod (kaolin.physics.simplicits.network.SkinningModule): Skinning field
            :math:`W_\theta`.
        rest_pts (torch.Tensor): Rest positions to decode at, of shape :math:`(\text{num_pts}, 3)`.
        qr_transform (torch.Tensor, optional): Reduced-coordinate reparameterization
            :math:`z = K z'`; must be the same one the driving :class:`~.reduced_model.ReducedModel`
            uses. Default: None.
    """

    def __init__(self, skinning_mod, rest_pts, qr_transform=None):
        self.rest_pts = rest_pts
        weights = skinning_mod.compute_skinning_weights(rest_pts)
        lbs = dense_lbs_matrix(rest_pts, weights)
        if qr_transform is not None:
            lbs = lbs @ qr_transform
        self.lbs = lbs

    def positions(self, reduced_coords):
        r"""Decode one reduced state to positions.

        Args:
            reduced_coords (torch.Tensor): Reduced coordinates :math:`z`,
                of shape :math:`(12 \text{num_handles},)`.

        Returns:
            torch.Tensor: Positions, of shape :math:`(\text{num_pts}, 3)`.
        """
        return self.rest_pts + (self.lbs @ reduced_coords).reshape(-1, 3)

    def trajectory(self, reduced_traj):
        r"""Decode a sequence of reduced states.

        Args:
            reduced_traj (list of torch.Tensor): Reduced coordinates per frame, each of shape
                :math:`(12 \text{num_handles},)`.

        Returns:
            torch.Tensor: Positions, of shape
            :math:`(\text{num_frames}, \text{num_pts}, 3)`.
        """
        return torch.stack([self.positions(z) for z in reduced_traj], dim=0)


def rollout(model, timestep, horizon, reduced_coords=None, reduced_velocity=None,
            num_newton_steps=4, line_search=True, hessian_regularizer=0.0, bptt_window=None,
            on_step=None):
    r"""Integrate the reduced dynamics for ``horizon`` steps, keeping the temporal graph intact.

    The velocity update is the backward-Euler one, :math:`\dot{z}_{t+1} = (z_{t+1} - z_t)/\Delta t`,
    and it is *not* detached: gradients propagate along the trajectory as well as through each
    solve. Set ``bptt_window`` to cut that chain at fixed intervals when memory or gradient
    conditioning demands it.

    Args:
        model (ReducedModel): The reduced system.
        timestep (float): Time step :math:`\Delta t` (in :math:`s`).
        horizon (int): Number of steps to integrate.
        reduced_coords (torch.Tensor, optional): Initial coordinates. Default: rest state (zeros).
        reduced_velocity (torch.Tensor, optional): Initial reduced velocity. Default: zeros.
        num_newton_steps (int, optional): Unrolled Newton iterations per step. Default: 4.
        line_search (bool, optional): Whether to line-search each Newton direction. Default: True.
        hessian_regularizer (float, optional): Passed to :func:`~.step.newton_step_unrolled`.
            Default: 0.0.
        bptt_window (int, optional): If given, detach the state every ``bptt_window`` steps so
            backpropagation spans at most that many steps. Default: None (full BPTT).
        on_step (callable, optional): Called as ``on_step(step_index, model)`` before each step,
            for time-varying controls (e.g. moving pins or a ramped load). Default: None.

    Returns:
        (list of torch.Tensor, list of torch.Tensor):
        the coordinates :math:`z_1 \ldots z_T` and the reduced velocities
        :math:`\dot{z}_1 \ldots \dot{z}_T`, each entry of shape :math:`(12 \text{num_handles},)`.
    """
    coords = model.zeros() if reduced_coords is None else reduced_coords
    velocity = model.zeros() if reduced_velocity is None else reduced_velocity

    coords_traj = []
    velocity_traj = []
    for step in range(horizon):
        if on_step is not None:
            on_step(step, model)
        coords_next = newton_step_unrolled(model, coords, velocity, timestep,
                                           num_newton_steps=num_newton_steps,
                                           line_search=line_search,
                                           hessian_regularizer=hessian_regularizer)
        velocity = (coords_next - coords) / timestep
        coords = coords_next
        coords_traj.append(coords)
        velocity_traj.append(velocity)

        if bptt_window is not None and (step + 1) % bptt_window == 0 and step + 1 < horizon:
            coords = coords.detach()
            velocity = velocity.detach()

    return coords_traj, velocity_traj
