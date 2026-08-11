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

r"""Simulation-in-the-loop training for Simplicits (research code).

``kaolin.physics.simplicits`` trains the neural skinning field :math:`W_\theta` with a *data-free*
objective: randomized affine handle configurations should be cheap in elastic energy, and the
weight columns should stay near-orthogonal. Nothing in that objective mentions the simulator, so
nothing forces the resulting reduced dynamics to agree with a full-order solve.

This package trains :math:`W_\theta` through the reduced simulator instead, so the loss is
"does the reduced trajectory match the full-order one". That requires an unbroken gradient path

.. math::

    \theta \to (W, \nabla_X W) \to \left(B, \tfrac{\partial F}{\partial z}, B^T M B\right)
        \to \{z_t\} \to \{\hat{x}_t\} \to L

which the production forward path cannot provide: it assembles its operators as
``warp.sparse`` matrices and runs its Newton solve under ``@torch.no_grad()``. So this package is
a **pure-PyTorch differentiable twin** of that solver, validated layer by layer against it
(see ``tests/python/kaolin/experimental/simplicits_diffsim/``). The Warp path remains the fast
forward path, the large-scale evaluator, and the correctness oracle.

Nothing here is imported by ``kaolin/__init__.py``; import the submodules explicitly.
"""

from . import data_gen
from . import dataset
from . import fd_check
from . import forces
from . import kinematics
from . import losses
from . import materials_torch
from . import projection
from . import reduced_model
from . import rollout
from . import step
from . import trainer

from .kinematics import (dense_dFdz_matrix, dense_lbs_matrix, qr_reparameterization,
                        reduced_mass_matrix, uniform_sample_volumes)
from .materials_torch import neohookean_energy, neohookean_gradient, neohookean_hessian
from .forces import Boundary, Floor, Gravity
from .reduced_model import ReducedModel, build_reduced_model
from .step import armijo_step_size, newton_step_unrolled
from .rollout import SkinningDecoder
from .fd_check import directional_fd_check

# NOTE: the ``rollout`` *function* is deliberately not re-exported here -- it would shadow the
# ``rollout`` submodule on this package. Use ``simplicits_diffsim.rollout.rollout``.

from .dataset import TrajectoryDataset, TrajectoryWindow
from .trainer import SimInLoopTrainer, TrainerConfig

__all__ = [
    'data_gen', 'dataset', 'fd_check', 'forces', 'kinematics', 'losses', 'materials_torch',
    'projection', 'reduced_model', 'rollout', 'step', 'trainer',
    'TrajectoryDataset', 'TrajectoryWindow', 'SimInLoopTrainer', 'TrainerConfig',
    'dense_dFdz_matrix', 'dense_lbs_matrix', 'qr_reparameterization', 'reduced_mass_matrix',
    'uniform_sample_volumes',
    'neohookean_energy', 'neohookean_gradient', 'neohookean_hessian',
    'Boundary', 'Floor', 'Gravity',
    'ReducedModel', 'build_reduced_model',
    'armijo_step_size', 'newton_step_unrolled',
    'SkinningDecoder',
    'directional_fd_check',
]
