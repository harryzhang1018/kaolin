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

r"""Trajectory-matching losses, plus a hybrid with the data-free Simplicits objective.

The data-free objective (:func:`kaolin.physics.simplicits.losses.compute_losses`) trains
:math:`W_\theta` to make randomized affine handle configurations cheap in elastic energy, and to
keep the weight columns near-orthogonal. It knows nothing about the dynamics. The losses here
instead compare a reduced *rollout* against a full-order one, and the hybrid keeps a shrinking
amount of the data-free term as a regularizer -- it is the only thing constraining the field away
from the sampled trajectories.
"""

import torch

from kaolin.physics.simplicits.losses import loss_elastic

__all__ = [
    'position_loss',
    'velocity_loss',
    'defo_grad_loss',
    'trajectory_loss',
    'per_vertex_l2_error',
    'ortho_term',
    'data_free_regularizer',
]


def ortho_term(weights):
    r"""Orthogonality of the skinning weight columns, :math:`\|W^T W - I\|^2 / H^2`.

    Numerically identical to :func:`kaolin.physics.simplicits.losses.loss_ortho`, reimplemented
    only because that function builds its identity with ``torch.eye(..., device=...)`` and no
    ``dtype``, so it is float32 whatever ``weights`` is; in float64 the forward pass promotes but
    the backward pass raises ``Found dtype Float but expected Double``. Since the finite-difference
    gates in this package require float64, the term is rebuilt here rather than worked around.
    ``test_training_pipeline.py`` asserts the two agree in float32.

    Args:
        weights (torch.Tensor): Skinning weights, of shape
            :math:`(\text{num_samples}, \text{num_handles})`.

    Returns:
        torch.Tensor: Scalar loss.
    """
    gram = weights.transpose(0, 1) @ weights
    eye = torch.eye(gram.shape[0], device=weights.device, dtype=weights.dtype)
    return ((gram - eye) ** 2).mean()


def position_loss(pred_positions, target_positions, node_weights=None):
    r"""Mean squared position error over a trajectory.

    Args:
        pred_positions (torch.Tensor): Predicted positions, of shape
            :math:`(\text{num_frames}, \text{num_pts}, 3)`.
        target_positions (torch.Tensor): Full-order positions, same shape.
        node_weights (torch.Tensor, optional): Per-point weights, of shape
            :math:`(\text{num_pts},)`, e.g. nodal volumes for a mass-weighted norm.
            Default: None (uniform).

    Returns:
        torch.Tensor: Scalar loss.
    """
    sq = ((pred_positions - target_positions) ** 2).sum(-1)
    if node_weights is None:
        return sq.mean()
    return (sq * node_weights).sum() / (node_weights.sum() * sq.shape[0])


def velocity_loss(pred_positions, target_positions, timestep):
    r"""Mean squared error of the finite-difference velocities implied by two trajectories.

    Penalizing positions alone leaves the reduced model free to lag or lead the full-order motion
    within the position tolerance; matching velocities pins the phase.

    Args:
        pred_positions (torch.Tensor): Predicted positions, of shape
            :math:`(\text{num_frames}, \text{num_pts}, 3)`.
        target_positions (torch.Tensor): Full-order positions, same shape.
        timestep (float): Time step :math:`\Delta t` (in :math:`s`).

    Returns:
        torch.Tensor: Scalar loss, or zero if fewer than two frames are given.
    """
    if pred_positions.shape[0] < 2:
        return pred_positions.new_zeros(())
    pred_vel = (pred_positions[1:] - pred_positions[:-1]) / timestep
    target_vel = (target_positions[1:] - target_positions[:-1]) / timestep
    return ((pred_vel - target_vel) ** 2).sum(-1).mean()


def defo_grad_loss(pred_defo_grads, target_defo_grads):
    r"""Mean squared error between deformation gradients.

    A local-strain term: two trajectories can agree on positions at the sampled points while
    disagreeing on the strain field between them, which is what the elastic energy actually sees.

    Args:
        pred_defo_grads (torch.Tensor): Predicted deformation gradients, of shape
            :math:`(\text{num_frames}, \text{num_pts}, 3, 3)`.
        target_defo_grads (torch.Tensor): Full-order deformation gradients, same shape.

    Returns:
        torch.Tensor: Scalar loss.
    """
    return ((pred_defo_grads - target_defo_grads) ** 2).sum((-2, -1)).mean()


def trajectory_loss(pred_positions, target_positions, timestep, pred_defo_grads=None,
                    target_defo_grads=None, node_weights=None, pos_coeff=1.0, vel_coeff=0.0,
                    defo_grad_coeff=0.0):
    r"""Weighted sum of the position, velocity and deformation-gradient trajectory terms.

    Args:
        pred_positions (torch.Tensor): Predicted positions, of shape
            :math:`(\text{num_frames}, \text{num_pts}, 3)`.
        target_positions (torch.Tensor): Full-order positions, same shape.
        timestep (float): Time step :math:`\Delta t` (in :math:`s`).
        pred_defo_grads (torch.Tensor, optional): Predicted deformation gradients, of shape
            :math:`(\text{num_frames}, \text{num_pts}, 3, 3)`. Default: None.
        target_defo_grads (torch.Tensor, optional): Full-order deformation gradients, same shape.
            Default: None.
        node_weights (torch.Tensor, optional): Per-point weights for the position term.
            Default: None.
        pos_coeff (float, optional): Position term weight. Default: 1.0.
        vel_coeff (float, optional): Velocity term weight. Default: 0.0.
        defo_grad_coeff (float, optional): Deformation-gradient term weight. Default: 0.0.

    Returns:
        (torch.Tensor, dict): the total loss and a dict of the individual terms as floats, for
        logging.
    """
    pos = position_loss(pred_positions, target_positions, node_weights)
    total = pos_coeff * pos
    terms = {'position': float(pos)}

    if vel_coeff != 0.0:
        vel = velocity_loss(pred_positions, target_positions, timestep)
        total = total + vel_coeff * vel
        terms['velocity'] = float(vel)

    if defo_grad_coeff != 0.0:
        if pred_defo_grads is None or target_defo_grads is None:
            raise ValueError('defo_grad_coeff is nonzero but deformation gradients were not given')
        dfg = defo_grad_loss(pred_defo_grads, target_defo_grads)
        total = total + defo_grad_coeff * dfg
        terms['defo_grad'] = float(dfg)

    return total, terms


@torch.no_grad()
def per_vertex_l2_error(pred_positions, target_positions):
    r"""Per-vertex L2 distance between two trajectories, as a reporting metric.

    This is the quantity ``run_l2_error_regression_test`` in
    ``tests/python/kaolin/physics/simplicits/test_simplicits_training_sim.py`` thresholds, so
    reported numbers stay comparable with the existing regression tests.

    Args:
        pred_positions (torch.Tensor): Predicted positions, of shape
            :math:`(\text{num_frames}, \text{num_pts}, 3)`.
        target_positions (torch.Tensor): Full-order positions, same shape.

    Returns:
        torch.Tensor: Distances, of shape :math:`(\text{num_frames}, \text{num_pts})`.
    """
    return (pred_positions - target_positions).norm(dim=-1)


def data_free_regularizer(skinning_mod, pts, yms, prs, rhos, appx_vol, num_samples=1000,
                          batch_size=10, interp_step=1.0, elastic_coeff=1.0, ortho_coeff=1.0,
                          generator=None):
    r"""The original data-free Simplicits objective, reused verbatim as a regularizer.

    Calls :func:`kaolin.physics.simplicits.losses.loss_elastic` directly rather than
    ``compute_losses``, so the sampling can be controlled (and seeded) here. The orthogonality term
    comes from :func:`ortho_term` for the dtype reason documented there. Note that, exactly as
    upstream, both terms see the network's raw output -- ``num_handles - 1`` learned columns, with
    no constant handle -- while the simulator uses all ``num_handles``.

    Note the deliberate inconsistency this inherits: ``loss_elastic`` evaluates the Neo-Hookean
    energy with ``reparameterize_lame=False`` and obtains :math:`F` by finite differences, while
    the simulator -- and therefore :mod:`.materials_torch` -- uses ``True`` and an analytic
    :math:`\nabla_X W`. The upstream function is left untouched so published weights and the
    reference training logs stay valid; only the twin is self-consistent.

    Args:
        skinning_mod (kaolin.physics.simplicits.network.SkinningModule): Skinning field.
        pts (torch.Tensor): Normalized sample points, of shape :math:`(\text{num_pts}, 3)`.
        yms (torch.Tensor): Point-wise Young's modulus, of shape :math:`(\text{num_pts},)`.
        prs (torch.Tensor): Point-wise Poisson's ratio, of shape :math:`(\text{num_pts},)`.
        rhos (torch.Tensor): Point-wise density, of shape :math:`(\text{num_pts},)`.
        appx_vol (float): Approximate object volume (in :math:`m^3`).
        num_samples (int, optional): Points to sample per call. Default: 1000.
        batch_size (int, optional): Number of random handle configurations. Default: 10.
        interp_step (float, optional): Linear-to-Neo-Hookean interpolation, 1.0 being fully
            Neo-Hookean. Default: 1.0.
        elastic_coeff (float, optional): Elastic term weight. Default: 1.0.
        ortho_coeff (float, optional): Orthogonality term weight. Default: 1.0.
        generator (torch.Generator, optional): Generator for the sampling. Default: None.

    Returns:
        (torch.Tensor, torch.Tensor): the elastic and orthogonality terms.
    """
    indices = torch.randint(low=0, high=pts.shape[0], size=(num_samples,), device=pts.device,
                            generator=generator)
    sample_pts = pts[indices]
    weights = skinning_mod(sample_pts)
    transforms = 0.1 * torch.randn(batch_size, weights.shape[-1], 3, 4, dtype=pts.dtype,
                                   device=pts.device, generator=generator)
    # loss_elastic broadcasts the Lame parameters as ``mus.expand(num_samples, batch_size)``, which
    # requires a trailing singleton dimension; ``training.train_step`` unsqueezes for the same
    # reason. This module's own convention is (num_pts,), so convert here.
    elastic = elastic_coeff * loss_elastic(skinning_mod, sample_pts, yms[indices].unsqueeze(-1),
                                           prs[indices].unsqueeze(-1),
                                           rhos[indices].unsqueeze(-1), transforms, appx_vol,
                                           interp_step)
    ortho = ortho_coeff * ortho_term(weights)
    return elastic, ortho
