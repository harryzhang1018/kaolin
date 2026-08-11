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

r"""A differentiable twin of the Simplicits reduced-order backward-Euler system.

:class:`ReducedModel` assembles the same operators as ``SimplicitsScene`` -- :math:`B`,
:math:`\partial F / \partial z`, :math:`B^T M B` -- and evaluates the same backward-Euler residual
and Hessian, but as pure PyTorch expressions of the skinning weights. Because the weights come
from :meth:`kaolin.physics.simplicits.network.SkinningModule.compute_skinning_weights` and
:meth:`~kaolin.physics.simplicits.network.SkinningModule.compute_dwdx`, the whole chain
:math:`\theta \to (W, \nabla_X W) \to (B, \partial F/\partial z, B^T M B) \to g, H` is
differentiable, which is what simulation-in-the-loop training needs.

The Warp scene remains the fast forward path and the correctness oracle; the equivalence tests in
``tests/python/kaolin/experimental/simplicits_diffsim/`` compare the two layer by layer.
"""

import copy

import torch

from kaolin.physics.utils.torch_utilities import hess_reduction

from .kinematics import dense_dFdz_matrix, dense_lbs_matrix, qr_reparameterization, reduced_mass_matrix
from .materials_torch import neohookean_energy, neohookean_gradient, neohookean_hessian

__all__ = [
    'ReducedModel',
    'build_reduced_model',
]


class ReducedModel:
    r"""Reduced-order elastodynamic system for a single object, differentiable in the skinning
    weights.

    The reduced coordinate :math:`z` stacks the :math:`3 \times 4` affine handle transforms, so
    positions are :math:`x = X + Bz` and deformation gradients are
    :math:`F = I + \text{reshape}\left(\frac{\partial F}{\partial z} z\right)`.

    Args:
        pts (torch.Tensor): Rest positions of the quadrature points,
            of shape :math:`(\text{num_samples}, 3)`.
        weights (torch.Tensor): Skinning weights including the constant handle,
            of shape :math:`(\text{num_samples}, \text{num_handles})`.
        weights_jac (torch.Tensor): Spatial gradient of the skinning weights,
            of shape :math:`(\text{num_samples}, \text{num_handles}, 3)`.
        mus (torch.Tensor): Lame :math:`\mu`, of shape :math:`(\text{num_samples},)`.
        lams (torch.Tensor): Lame :math:`\lambda`, of shape :math:`(\text{num_samples},)`.
        vols (torch.Tensor): Integration volumes, of shape :math:`(\text{num_samples},)`.
        masses (torch.Tensor): Per-point masses, of shape :math:`(\text{num_samples},)`.
        pt_forces (list, optional): Point-wise potentials from :mod:`.forces`, each exposing
            ``energy``, ``gradient`` and ``hessian_blocks``. Default: None.
        qr_transform (torch.Tensor, optional): Reduced-coordinate reparameterization
            :math:`z = K z'`, of shape :math:`(12 \text{num_handles}, 12 \text{num_handles})`.
            Pass ``'auto'`` to compute it from :math:`B` via :func:`.qr_reparameterization`.
            Default: None.
        reparameterize_lame (bool, optional): If True, use :math:`\lambda \leftarrow \lambda + \mu`
            in the Neo-Hookean energy, matching ``SimplicitsScene``. Note the data-free
            ``kaolin.physics.simplicits.losses.loss_elastic`` uses False; the twin deliberately
            follows the simulator instead. Default: True.
    """

    def __init__(self, pts, weights, weights_jac, mus, lams, vols, masses,
                 pt_forces=None, qr_transform=None, reparameterize_lame=True):
        self.pts = pts
        self.mus = mus
        self.lams = lams
        self.vols = vols
        self.masses = masses
        self.pt_forces = list(pt_forces) if pt_forces is not None else []
        self.reparameterize_lame = reparameterize_lame
        self.num_samples = pts.shape[0]
        self.num_handles = weights.shape[1]

        lbs = dense_lbs_matrix(pts, weights)
        dfdz = dense_dFdz_matrix(pts, weights, weights_jac)
        if isinstance(qr_transform, str):
            if qr_transform != 'auto':
                raise ValueError(f"qr_transform must be a tensor, None or 'auto', got {qr_transform!r}")
            qr_transform = qr_reparameterization(lbs)
        self.qr_transform = qr_transform
        if qr_transform is not None:
            lbs = lbs @ qr_transform
            dfdz = dfdz @ qr_transform

        self.lbs = lbs
        self.dFdz = dfdz
        self.reduced_mass = reduced_mass_matrix(lbs, masses)
        self.eye3 = torch.eye(3, device=pts.device, dtype=pts.dtype)

    @property
    def num_reduced_dofs(self):
        r"""int: Size of the reduced coordinate vector, :math:`12 \times \text{num_handles}`."""
        return self.lbs.shape[1]

    def with_pt_forces(self, pt_forces):
        r"""A view of this model carrying a different set of point-wise forces.

        :math:`B`, :math:`\partial F/\partial z` and :math:`B^T M B` depend on the skinning field
        and the masses, never on the loads, so several load scenarios can share one assembly and
        one network evaluation. The returned model **shares** those tensors -- including their
        autograd history -- so a gradient taken through it still reaches :math:`\theta`.

        Args:
            pt_forces (list): Point-wise potentials from :mod:`.forces`.

        Returns:
            ReducedModel: A shallow copy with ``pt_forces`` replaced.
        """
        other = copy.copy(self)
        other.pt_forces = list(pt_forces)
        return other

    def zeros(self):
        r"""Zero reduced coordinate vector (the rest state).

        Returns:
            torch.Tensor: Zeros, of shape :math:`(12 \text{num_handles},)`.
        """
        return self.lbs.new_zeros(self.num_reduced_dofs)

    def positions(self, reduced_coords):
        r"""Deformed sample positions :math:`x = X + Bz`.

        Args:
            reduced_coords (torch.Tensor): Reduced coordinates :math:`z`,
                of shape :math:`(12 \text{num_handles},)`.

        Returns:
            torch.Tensor: Positions, of shape :math:`(\text{num_samples}, 3)`.
        """
        return self.pts + (self.lbs @ reduced_coords).reshape(self.num_samples, 3)

    def defo_grads(self, reduced_coords):
        r"""Deformation gradients :math:`F = I + \text{reshape}(\partial F/\partial z \, z)`.

        Args:
            reduced_coords (torch.Tensor): Reduced coordinates :math:`z`,
                of shape :math:`(12 \text{num_handles},)`.

        Returns:
            torch.Tensor: Deformation gradients, of shape :math:`(\text{num_samples}, 3, 3)`.
        """
        flat = self.dFdz @ reduced_coords
        return flat.reshape(self.num_samples, 3, 3) + self.eye3

    def potential_energy(self, reduced_coords):
        r"""Total potential energy: elastic strain energy plus every point-wise force.

        Args:
            reduced_coords (torch.Tensor): Reduced coordinates :math:`z`,
                of shape :math:`(12 \text{num_handles},)`.

        Returns:
            torch.Tensor: Scalar energy.
        """
        energy = neohookean_energy(self.defo_grads(reduced_coords), self.mus, self.lams, self.vols,
                                   self.reparameterize_lame).sum()
        if self.pt_forces:
            positions = self.positions(reduced_coords)
            for force in self.pt_forces:
                energy = energy + force.energy(positions)
        return energy

    def potential_gradient(self, reduced_coords):
        r"""Analytic reduced gradient
        :math:`\partial E / \partial z = B^T \partial E/\partial x
        + \left(\partial F/\partial z\right)^T \partial E/\partial F`.

        Computed in closed form rather than by ``autograd.grad`` on :meth:`potential_energy`, so
        that differentiating the assembled residual with respect to :math:`\theta` stays a
        first-order backward pass.

        Args:
            reduced_coords (torch.Tensor): Reduced coordinates :math:`z`,
                of shape :math:`(12 \text{num_handles},)`.

        Returns:
            torch.Tensor: Gradient, of shape :math:`(12 \text{num_handles},)`.
        """
        dedf = neohookean_gradient(self.defo_grads(reduced_coords), self.mus, self.lams, self.vols,
                                   self.reparameterize_lame)
        grad = self.dFdz.transpose(0, 1) @ dedf.reshape(-1)
        if self.pt_forces:
            positions = self.positions(reduced_coords)
            dedx = sum(force.gradient(positions) for force in self.pt_forces)
            grad = grad + self.lbs.transpose(0, 1) @ dedx.reshape(-1)
        return grad

    def potential_hessian(self, reduced_coords):
        r"""Analytic reduced Hessian
        :math:`\partial^2 E / \partial z^2 = B^T H_x B
        + \left(\partial F/\partial z\right)^T H_F \left(\partial F/\partial z\right)`.

        Args:
            reduced_coords (torch.Tensor): Reduced coordinates :math:`z`,
                of shape :math:`(12 \text{num_handles},)`.

        Returns:
            torch.Tensor: Hessian, of shape
            :math:`(12 \text{num_handles}, 12 \text{num_handles})`.
        """
        blocks = neohookean_hessian(self.defo_grads(reduced_coords), self.mus, self.lams, self.vols,
                                    self.reparameterize_lame)
        hessian = hess_reduction(self.dFdz, blocks)
        if self.pt_forces:
            positions = self.positions(reduced_coords)
            pt_blocks = sum(force.hessian_blocks(positions) for force in self.pt_forces)
            hessian = hessian + hess_reduction(self.lbs, pt_blocks)
        return hessian

    def newton_energy(self, reduced_coords, reduced_coords_prev, reduced_velocity, timestep):
        r"""Backward-Euler incremental potential,
        :math:`\frac{1}{2} d^T (B^T M B) d + \Delta t^2 E_{\text{pot}}(z)` with
        :math:`d = z - z_{\text{prev}} - \Delta t \, \dot{z}`. This is the objective the Newton
        line search decreases.

        Args:
            reduced_coords (torch.Tensor): Trial coordinates :math:`z`,
                of shape :math:`(12 \text{num_handles},)`.
            reduced_coords_prev (torch.Tensor): Previous-step coordinates,
                of shape :math:`(12 \text{num_handles},)`.
            reduced_velocity (torch.Tensor): Previous-step reduced velocity,
                of shape :math:`(12 \text{num_handles},)`.
            timestep (float): Time step :math:`\Delta t` (in :math:`s`).

        Returns:
            torch.Tensor: Scalar energy.
        """
        delta = reduced_coords - reduced_coords_prev - timestep * reduced_velocity
        inertia = 0.5 * (delta @ (self.reduced_mass @ delta))
        return inertia + timestep * timestep * self.potential_energy(reduced_coords)

    def residual(self, reduced_coords, reduced_coords_prev, reduced_velocity, timestep):
        r"""Backward-Euler residual
        :math:`g = B^T M B \left(z - z_{\text{prev}} - \Delta t \dot{z}\right)
        + \Delta t^2 \, \partial E_{\text{pot}} / \partial z`.

        Matches ``SimplicitsScene._newton_G``.

        Args:
            reduced_coords (torch.Tensor): Trial coordinates :math:`z`,
                of shape :math:`(12 \text{num_handles},)`.
            reduced_coords_prev (torch.Tensor): Previous-step coordinates,
                of shape :math:`(12 \text{num_handles},)`.
            reduced_velocity (torch.Tensor): Previous-step reduced velocity,
                of shape :math:`(12 \text{num_handles},)`.
            timestep (float): Time step :math:`\Delta t` (in :math:`s`).

        Returns:
            torch.Tensor: Residual, of shape :math:`(12 \text{num_handles},)`.
        """
        delta = reduced_coords - reduced_coords_prev - timestep * reduced_velocity
        return self.reduced_mass @ delta + timestep * timestep * self.potential_gradient(reduced_coords)

    def hessian(self, reduced_coords, timestep):
        r"""Jacobian of :meth:`residual` with respect to :math:`z`,
        :math:`H = B^T M B + \Delta t^2 \, \partial^2 E_{\text{pot}} / \partial z^2`.

        Matches ``SimplicitsScene._newton_H`` with ``newton_hessian_regularizer=0``. The
        regularizer is deliberately *not* folded in here: it belongs to the linear solve, and an
        implicit-function (adjoint) derivative must use the unregularized Jacobian.

        Args:
            reduced_coords (torch.Tensor): Trial coordinates :math:`z`,
                of shape :math:`(12 \text{num_handles},)`.
            timestep (float): Time step :math:`\Delta t` (in :math:`s`).

        Returns:
            torch.Tensor: Hessian, of shape
            :math:`(12 \text{num_handles}, 12 \text{num_handles})`.
        """
        return self.reduced_mass + timestep * timestep * self.potential_hessian(reduced_coords)


def build_reduced_model(skinning_mod, pts, mus, lams, vols, masses, pt_forces=None,
                        qr_transform=None, reparameterize_lame=True):
    r"""Build a :class:`ReducedModel` by evaluating a skinning network at the quadrature points.

    The returned model holds :math:`B` and :math:`\partial F / \partial z` as tensors that are
    live functions of the network parameters, so it must be rebuilt after every optimizer step.

    Args:
        skinning_mod (kaolin.physics.simplicits.network.SkinningModule): Skinning field
            :math:`W_\theta`.
        pts (torch.Tensor): Rest positions of the quadrature points,
            of shape :math:`(\text{num_samples}, 3)`.
        mus (torch.Tensor): Lame :math:`\mu`, of shape :math:`(\text{num_samples},)`.
        lams (torch.Tensor): Lame :math:`\lambda`, of shape :math:`(\text{num_samples},)`.
        vols (torch.Tensor): Integration volumes, of shape :math:`(\text{num_samples},)`.
        masses (torch.Tensor): Per-point masses, of shape :math:`(\text{num_samples},)`.
        pt_forces (list, optional): Point-wise potentials from :mod:`.forces`. Default: None.
        qr_transform (torch.Tensor or str, optional): See :class:`ReducedModel`. Default: None.
        reparameterize_lame (bool, optional): See :class:`ReducedModel`. Default: True.

    Returns:
        ReducedModel: The assembled differentiable reduced system.
    """
    weights = skinning_mod.compute_skinning_weights(pts)
    weights_jac = skinning_mod.compute_dwdx(pts)
    return ReducedModel(pts, weights, weights_jac, mus, lams, vols, masses, pt_forces=pt_forces,
                        qr_transform=qr_transform, reparameterize_lame=reparameterize_lame)
