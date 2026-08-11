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

r"""Finite-difference verification of parameter gradients through a whole rollout.

A directional check rather than a full ``gradcheck``: a random unit direction :math:`d` in
flattened parameter space, compared against a central difference of the loss. One scalar
comparison per :math:`\epsilon`, so the cost is two extra forward rollouts instead of one per
parameter.

Two hazards this harness exists to avoid:

* **Never wrap the loss in** ``torch.no_grad()``. It silently changes what some functorch
  transforms return (``torch.func.hessian`` loses its inner ``jacrev``), so a "reference" computed
  that way can disagree with the analytic gradient by percent-level amounts while both are
  self-consistently wrong. The evaluations below run with grad tracking enabled and simply discard
  the graph.
* **Run in float64.** In float32 the central difference has no usable window: truncation error at
  large :math:`\epsilon` and cancellation at small :math:`\epsilon` never leave a plateau.
"""

import torch

__all__ = [
    'directional_fd_check',
]


def directional_fd_check(loss_fn, parameters, eps_sweep=(1e-4, 1e-5, 1e-6), direction=None,
                         generator=None, verbose=False):
    r"""Compare the analytic directional derivative of ``loss_fn`` against a central difference.

    Args:
        loss_fn (callable): Zero-argument callable returning a scalar tensor. It must re-read the
            parameters on every call (i.e. rebuild any cached :math:`B` /
            :math:`\partial F/\partial z`), otherwise the perturbed evaluations are meaningless.
        parameters (iterable of torch.Tensor): The parameters to differentiate with respect to,
            typically ``list(model.parameters())``.
        eps_sweep (tuple of float, optional): Perturbation magnitudes to try.
            Default: ``(1e-4, 1e-5, 1e-6)``.
        direction (torch.Tensor, optional): Unit direction in flattened parameter space. Default:
            a fresh random direction.
        generator (torch.Generator, optional): Generator for the random direction. Default: None.
        verbose (bool, optional): If True, print the per-:math:`\epsilon` table. Default: False.

    Returns:
        (float, float, dict): the best relative error over the sweep, the analytic directional
        derivative, and a dict mapping :math:`\epsilon` to the finite-difference estimate.
    """
    parameters = list(parameters)

    loss = loss_fn()
    grads = torch.autograd.grad(loss, parameters, allow_unused=True)
    flat_grad = torch.cat([torch.zeros_like(p).reshape(-1) if g is None else g.reshape(-1)
                           for p, g in zip(parameters, grads)])

    flat_params = torch.nn.utils.parameters_to_vector(parameters).detach().clone()
    if direction is None:
        direction = torch.empty_like(flat_params).normal_(generator=generator)
    direction = direction / direction.norm()

    analytic = (flat_grad * direction).sum().item()

    def loss_at(vec):
        torch.nn.utils.vector_to_parameters(vec, parameters)
        # Deliberately NOT under torch.no_grad(); see the module docstring.
        return loss_fn().detach().item()

    estimates = {}
    try:
        for eps in eps_sweep:
            plus = loss_at(flat_params + eps * direction)
            minus = loss_at(flat_params - eps * direction)
            estimates[eps] = (plus - minus) / (2.0 * eps)
    finally:
        torch.nn.utils.vector_to_parameters(flat_params, parameters)

    denom = max(abs(analytic), 1e-30)
    errors = {eps: abs(fd - analytic) / denom for eps, fd in estimates.items()}
    best = min(errors.values())

    if verbose:
        print(f'analytic directional derivative: {analytic:.14e}')
        print(f'{"eps":>10} {"central difference":>24} {"rel err":>12}')
        for eps in estimates:
            print(f'{eps:10.1e} {estimates[eps]:24.14e} {errors[eps]:12.3e}')

    return best, analytic, estimates
