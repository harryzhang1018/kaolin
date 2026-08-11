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

r"""Stage A: a differentiable backward-Euler step by unrolling Newton's method.

A fixed number of Newton iterations is executed *inside* the autograd graph, so the returned
:math:`z_{t+1}` is a differentiable function of the skinning weights through every residual,
Hessian and linear solve. Only the line-search step size -- a scalar chosen by a
non-differentiable search -- is detached; that is the standard treatment, since :math:`\alpha`
enters as a piecewise-constant function whose derivative carries no useful signal.

What must *not* be detached, and is enforced by ``test_gradients.py``: the residual :math:`g`,
the Hessian :math:`H`, :math:`B`, :math:`\partial F/\partial z`, :math:`B^T M B`, and the state
recurrence :math:`z_t \to z_{t+1}` inside a training window.
"""

import torch

__all__ = [
    'armijo_step_size',
    'newton_step_unrolled',
]


@torch.no_grad()
def armijo_step_size(model, reduced_coords, reduced_coords_prev, reduced_velocity, timestep,
                     direction, gradient, initial_step_size=1.0, armijo_alpha=1e-3,
                     backtrack_beta=0.6, max_steps=10):
    r"""Backtracking line-search step size for one Newton direction, returned as a scalar.

    This replicates :func:`kaolin.physics.common.optimization._line_search` with unit bounds,
    including two of its quirks, so that rollouts of the twin track the production solver:

    * on the *first* iterate satisfying the Armijo condition the step is **grown** by
      :math:`1/\beta` and retried; only a second success returns. A grown step that then fails
      keeps shrinking from there.
    * if ``max_steps`` is exhausted the last trial step is returned whether or not it satisfies
      the Armijo condition.

    Bounds are unit (there is no collision-bound machinery in the twin yet), so the effective
    step size is capped at 1.

    Args:
        model (ReducedModel): The reduced system.
        reduced_coords (torch.Tensor): Current iterate :math:`z`,
            of shape :math:`(12 \text{num_handles},)`.
        reduced_coords_prev (torch.Tensor): Previous-step coordinates,
            of shape :math:`(12 \text{num_handles},)`.
        reduced_velocity (torch.Tensor): Previous-step reduced velocity,
            of shape :math:`(12 \text{num_handles},)`.
        timestep (float): Time step :math:`\Delta t` (in :math:`s`).
        direction (torch.Tensor): Newton direction :math:`\Delta z`,
            of shape :math:`(12 \text{num_handles},)`.
        gradient (torch.Tensor): Residual at ``reduced_coords``,
            of shape :math:`(12 \text{num_handles},)`.
        initial_step_size (float, optional): Initial :math:`t`. Default: 1.0.
        armijo_alpha (float, optional): Sufficient-decrease constant. Default: 1e-3.
        backtrack_beta (float, optional): Backtracking factor. Default: 0.6.
        max_steps (int, optional): Maximum trial steps. Default: 10.

    Returns:
        float: Step size :math:`\alpha \in (0, 1]`.
    """
    step = initial_step_size
    energy = model.newton_energy(reduced_coords, reduced_coords_prev, reduced_velocity, timestep)
    can_break = False
    scale = min(1.0, step)

    for _ in range(max_steps):
        trial = reduced_coords + scale * direction
        energy_new = model.newton_energy(trial, reduced_coords_prev, reduced_velocity, timestep)
        if energy_new <= energy + armijo_alpha * scale * (gradient @ direction):
            if can_break:
                return scale
            can_break = True
            step = step / backtrack_beta
        else:
            step = step * backtrack_beta
        scale = min(1.0, step)

    return scale


def newton_step_unrolled(model, reduced_coords_prev, reduced_velocity, timestep,
                         num_newton_steps=4, line_search=True, hessian_regularizer=0.0,
                         conv_tol=0.0):
    r"""One differentiable backward-Euler step, by unrolling ``num_newton_steps`` Newton
    iterations inside the autograd graph.

    Starting from the inertial guess :math:`z^{(0)} = z_t + \Delta t \, \dot{z}_t`, each iteration
    solves :math:`H \Delta z = -g` densely and takes a (possibly line-searched) step. Iteration
    count is fixed rather than tolerance-driven by default, so the graph -- and therefore the
    gradient -- has the same shape for every sample in a batch.

    Args:
        model (ReducedModel): The reduced system. Its :math:`B` and :math:`\partial F/\partial z`
            must still be attached to the network parameters.
        reduced_coords_prev (torch.Tensor): Coordinates at time :math:`t`,
            of shape :math:`(12 \text{num_handles},)`.
        reduced_velocity (torch.Tensor): Reduced velocity at time :math:`t`,
            of shape :math:`(12 \text{num_handles},)`.
        timestep (float): Time step :math:`\Delta t` (in :math:`s`).
        num_newton_steps (int, optional): Number of unrolled iterations :math:`K`. Default: 4.
        line_search (bool, optional): If True, scale each direction by
            :func:`armijo_step_size`; if False, take full Newton steps. Default: True.
        hessian_regularizer (float, optional): Adds ``hessian_regularizer * I`` to :math:`H`
            before the solve only, mirroring ``SimplicitsScene.newton_hessian_regularizer``.
            Default: 0.0.
        conv_tol (float, optional): If positive, stop early once
            :math:`|\Delta z \cdot g| < \text{conv_tol}`, matching the production convergence
            test. Leave at 0 for training so the unrolled depth is fixed. Default: 0.0.

    Returns:
        torch.Tensor: Coordinates at time :math:`t + \Delta t`,
        of shape :math:`(12 \text{num_handles},)`.
    """
    reduced_coords = reduced_coords_prev + timestep * reduced_velocity
    eye = None

    for _ in range(num_newton_steps):
        gradient = model.residual(reduced_coords, reduced_coords_prev, reduced_velocity, timestep)
        hessian = model.hessian(reduced_coords, timestep)
        if hessian_regularizer != 0.0:
            if eye is None:
                eye = torch.eye(hessian.shape[0], device=hessian.device, dtype=hessian.dtype)
            hessian = hessian + hessian_regularizer * eye
        direction = torch.linalg.solve(hessian, -gradient)

        if conv_tol > 0.0 and torch.abs(direction @ gradient).item() < conv_tol:
            break

        if line_search:
            scale = armijo_step_size(model, reduced_coords.detach(), reduced_coords_prev.detach(),
                                     reduced_velocity.detach(), timestep, direction.detach(),
                                     gradient.detach())
        else:
            scale = 1.0
        reduced_coords = reduced_coords + scale * direction

    return reduced_coords
