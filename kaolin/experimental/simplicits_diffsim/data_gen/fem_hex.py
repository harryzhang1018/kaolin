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

r"""A trilinear-hexahedron full-order Neo-Hookean solver, used as ground truth.

Deliberately *not* built on ``warp.fem``: the constitutive law here is the very same Python
functions the reduced twin uses (:mod:`..materials_torch`), so the residual error between the two
is reduction error by construction, with no possibility of a material mismatch. Those functions
are themselves verified against the Warp kernels the production simulator runs, so the chain of
custody back to ``kaolin.physics`` is intact -- see
``tests/python/kaolin/experimental/simplicits_diffsim/test_materials_torch.py``.

The time integrator is the same backward-Euler form as
:meth:`..reduced_model.ReducedModel.residual`,

.. math::

    M (x - x_t - \Delta t \, v_t) + \Delta t^2 \frac{\partial E_{\text{pot}}}{\partial x} = 0,

so a full-order and a reduced trajectory differ only in the space they are solved in.

Elements are :math:`Q_1` hexahedra with :math:`2 \times 2 \times 2` Gauss quadrature, and boundary
conditions are enforced as hard Dirichlet constraints by eliminating the pinned degrees of freedom
(the reduced model can only approximate that, via its penalty term -- one of the error sources to
report rather than hide).
"""

import numpy as np
import torch

from ..materials_torch import neohookean_energy, neohookean_gradient, neohookean_hessian

__all__ = [
    'HexGrid',
    'FullOrderNeohookeanSolver',
]

_GAUSS_2PT = 1.0 / np.sqrt(3.0)
# Reference-cube corner signs, in the node order used for each element.
_CORNER_SIGNS = np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)],
                         dtype=np.float64)


class HexGrid:
    r"""A structured grid of trilinear hexahedra over an axis-aligned box.

    Args:
        bounds_min (sequence of float): Lower box corner, length 3 (in :math:`m`).
        bounds_max (sequence of float): Upper box corner, length 3 (in :math:`m`).
        resolution (sequence of int): Number of *elements* along each axis, length 3. The node
            count is ``prod(resolution + 1)``.
        device (torch.device, optional): Device for all tensors. Default: None.
        dtype (torch.dtype, optional): Precision. Default: ``torch.float64``.
    """

    def __init__(self, bounds_min, bounds_max, resolution, device=None, dtype=torch.float64):
        self.device = device
        self.dtype = dtype
        self.resolution = tuple(int(r) for r in resolution)
        self.bounds_min = torch.as_tensor(bounds_min, device=device, dtype=dtype)
        self.bounds_max = torch.as_tensor(bounds_max, device=device, dtype=dtype)

        nx, ny, nz = self.resolution
        axes = [torch.linspace(float(self.bounds_min[d]), float(self.bounds_max[d]),
                               self.resolution[d] + 1, device=device, dtype=dtype)
                for d in range(3)]
        grid = torch.meshgrid(*axes, indexing='ij')
        self.nodes = torch.stack([g.reshape(-1) for g in grid], dim=-1)
        self.num_nodes = self.nodes.shape[0]
        self.num_elements = nx * ny * nz

        self.elements = self._build_elements()
        self._build_quadrature()

    def _node_index(self, i, j, k):
        _, ny, nz = self.resolution
        return (i * (ny + 1) + j) * (nz + 1) + k

    def _build_elements(self):
        """Node indices of every element, ordered to match ``_CORNER_SIGNS``."""
        nx, ny, nz = self.resolution
        base_i, base_j, base_k = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz),
                                             indexing='ij')
        base_i, base_j, base_k = base_i.reshape(-1), base_j.reshape(-1), base_k.reshape(-1)
        corners = []
        for sign in _CORNER_SIGNS:
            offset = ((sign + 1) / 2).astype(np.int64)
            corners.append(self._node_index(base_i + offset[0], base_j + offset[1],
                                            base_k + offset[2]))
        return torch.as_tensor(np.stack(corners, axis=-1), device=self.device, dtype=torch.long)

    def _build_quadrature(self):
        r"""Per-element, per-quadrature-point shape-function gradients and volumes.

        Stores ``shape_grads`` of shape :math:`(E, Q, 8, 3)` -- that is
        :math:`\partial N_a / \partial X_m` -- and ``qp_volumes`` of shape :math:`(E, Q)`.
        """
        signs = torch.as_tensor(_CORNER_SIGNS, device=self.device, dtype=self.dtype)
        offsets = torch.as_tensor([[i, j, k] for i in (-_GAUSS_2PT, _GAUSS_2PT)
                                   for j in (-_GAUSS_2PT, _GAUSS_2PT)
                                   for k in (-_GAUSS_2PT, _GAUSS_2PT)],
                                  device=self.device, dtype=self.dtype)

        # (Q, 8): trilinear shape functions at each quadrature point.
        terms = 1.0 + offsets.unsqueeze(1) * signs.unsqueeze(0)
        self.shape_values = 0.125 * terms.prod(dim=-1)

        # (Q, 8, 3): d N_a / d xi_m
        grad_ref = torch.empty(offsets.shape[0], 8, 3, device=self.device, dtype=self.dtype)
        for axis in range(3):
            others = [d for d in range(3) if d != axis]
            grad_ref[..., axis] = 0.125 * signs[:, axis].unsqueeze(0) * \
                terms[..., others[0]] * terms[..., others[1]]

        # Reference-to-world Jacobian, per element and quadrature point.
        element_nodes = self.nodes[self.elements]
        jacobian = torch.einsum('eam,qan->eqmn', element_nodes, grad_ref)
        det = torch.linalg.det(jacobian)
        if float(det.min()) <= 0.0:
            raise ValueError('degenerate or inverted hexahedra in the grid')
        inv_jacobian = torch.linalg.inv(jacobian)

        # d N_a / d X_m = d N_a / d xi_n * d xi_n / d X_m
        self.shape_grads = torch.einsum('qan,eqnm->eqam', grad_ref, inv_jacobian)
        # Gauss weights are all 1 for the 2-point rule, so the volume is just |det J|.
        self.qp_volumes = det
        self.num_qp_per_element = offsets.shape[0]

    def element_volumes(self):
        r"""Total volume per element.

        Returns:
            torch.Tensor: Volumes, of shape :math:`(\text{num_elements},)`.
        """
        return self.qp_volumes.sum(dim=1)

    def total_volume(self):
        r"""Total mesh volume.

        Returns:
            torch.Tensor: Scalar volume.
        """
        return self.qp_volumes.sum()

    def lumped_masses(self, density):
        r"""Row-sum-lumped nodal masses, :math:`m_a = \rho \sum_q w_q N_a(q)`.

        Args:
            density (float): Uniform density (in :math:`kg/m^3`).

        Returns:
            torch.Tensor: Nodal masses, of shape :math:`(\text{num_nodes},)`.
        """
        contributions = density * torch.einsum('qa,eq->ea', self.shape_values, self.qp_volumes)
        masses = torch.zeros(self.num_nodes, device=self.device, dtype=self.dtype)
        return masses.index_add(0, self.elements.reshape(-1), contributions.reshape(-1))

    def nodal_volumes(self):
        r"""Row-sum-lumped nodal volumes, the ``density = 1`` case of :meth:`lumped_masses`.

        Returns:
            torch.Tensor: Nodal volumes, of shape :math:`(\text{num_nodes},)`.
        """
        return self.lumped_masses(1.0)

    def defo_grads(self, displacements):
        r"""Deformation gradient at every quadrature point.

        Args:
            displacements (torch.Tensor): Nodal displacements, of shape
                :math:`(\text{num_nodes}, 3)`.

        Returns:
            torch.Tensor: Deformation gradients, of shape
            :math:`(\text{num_elements}, Q, 3, 3)`.
        """
        element_disp = displacements[self.elements]
        grads = torch.einsum('eai,eqam->eqim', element_disp, self.shape_grads)
        return grads + torch.eye(3, device=self.device, dtype=self.dtype)

    def quadrature_points(self):
        r"""World positions of the quadrature points.

        Returns:
            torch.Tensor: Positions, of shape :math:`(\text{num_elements} \times Q, 3)`.
        """
        element_nodes = self.nodes[self.elements]
        return torch.einsum('qa,eam->eqm', self.shape_values, element_nodes).reshape(-1, 3)

    def surface_faces(self):
        r"""Triangulated boundary faces of the grid, for export and rendering.

        Returns:
            torch.Tensor: Triangle indices, of shape :math:`(\text{num_triangles}, 3)`.
        """
        nx, ny, nz = self.resolution
        quads = []

        def add(i_range, j_range, k_range, corner_offsets):
            grid = np.meshgrid(i_range, j_range, k_range, indexing='ij')
            base = [g.reshape(-1) for g in grid]
            face = [self._node_index(base[0] + off[0], base[1] + off[1], base[2] + off[2])
                    for off in corner_offsets]
            quads.append(np.stack(face, axis=-1))

        unit_i = np.arange(nx)
        unit_j = np.arange(ny)
        unit_k = np.arange(nz)
        add([0], unit_j, unit_k, [(0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)])
        add([nx], unit_j, unit_k, [(0, 0, 0), (0, 1, 0), (0, 1, 1), (0, 0, 1)])
        add(unit_i, [0], unit_k, [(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)])
        add(unit_i, [ny], unit_k, [(0, 0, 0), (0, 0, 1), (1, 0, 1), (1, 0, 0)])
        add(unit_i, unit_j, [0], [(0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)])
        add(unit_i, unit_j, [nz], [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)])

        quads = np.concatenate(quads, axis=0)
        triangles = np.concatenate([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]], axis=0)
        return torch.as_tensor(triangles, device=self.device, dtype=torch.long)


class FullOrderNeohookeanSolver:
    r"""Backward-Euler Neo-Hookean dynamics on a :class:`HexGrid`, with hard Dirichlet pins.

    Args:
        grid (HexGrid): The mesh.
        youngs_modulus (float): Young's modulus (in :math:`kg/m/s^2`).
        poisson_ratio (float): Poisson's ratio.
        density (float): Density (in :math:`kg/m^3`).
        gravity (sequence of float, optional): Acceleration vector, length 3. Follows the
            ``SimplicitsScene`` convention where the energy is :math:`+ m g \cdot x`, so
            ``(0, 9.8, 0)`` pulls along :math:`-y`. Default: ``(0.0, 9.8, 0.0)``.
        timestep (float, optional): Time step (in :math:`s`). Default: 0.05.
        pinned_nodes (torch.Tensor, optional): Indices of nodes held at their rest position.
            Default: None.
        reparameterize_lame (bool, optional): Passed through to the material, so it can be matched
            to the reduced model. Default: True.
    """

    def __init__(self, grid, youngs_modulus, poisson_ratio, density, gravity=(0.0, 9.8, 0.0),
                 timestep=0.05, pinned_nodes=None, reparameterize_lame=True,
                 linear_solver='auto', dense_threshold=4096, cg_tolerance=1e-12,
                 cg_max_iterations=20000):
        from kaolin.physics.materials.material_utils import to_lame

        self.grid = grid
        self.timestep = timestep
        self.density = density
        self.reparameterize_lame = reparameterize_lame
        self.linear_solver = linear_solver
        self.dense_threshold = dense_threshold
        self.cg_tolerance = cg_tolerance
        self.cg_max_iterations = cg_max_iterations
        device, dtype = grid.device, grid.dtype

        num_qp = grid.num_elements * grid.num_qp_per_element
        yms = torch.full((num_qp,), float(youngs_modulus), device=device, dtype=dtype)
        prs = torch.full((num_qp,), float(poisson_ratio), device=device, dtype=dtype)
        self.mus, self.lams = to_lame(yms, prs)
        self.qp_volumes = grid.qp_volumes.reshape(-1)

        self.masses = grid.lumped_masses(density)
        self.gravity = torch.as_tensor(gravity, device=device, dtype=dtype)

        self.pinned_nodes = pinned_nodes
        num_dofs = 3 * grid.num_nodes
        free = torch.ones(num_dofs, device=device, dtype=torch.bool)
        if pinned_nodes is not None and pinned_nodes.numel() > 0:
            pinned_dofs = (3 * pinned_nodes.unsqueeze(-1)
                           + torch.arange(3, device=device)).reshape(-1)
            free[pinned_dofs] = False
        self.free_dofs = torch.nonzero(free, as_tuple=False).squeeze(1)

    def potential_gradient(self, displacements):
        r"""Nodal gradient of the total potential energy.

        Args:
            displacements (torch.Tensor): Nodal displacements, of shape
                :math:`(\text{num_nodes}, 3)`.

        Returns:
            torch.Tensor: Gradient, of shape :math:`(\text{num_nodes}, 3)`.
        """
        grid = self.grid
        defo_grads = grid.defo_grads(displacements).reshape(-1, 3, 3)
        stress = neohookean_gradient(defo_grads, self.mus, self.lams, self.qp_volumes,
                                     self.reparameterize_lame)
        stress = stress.reshape(grid.num_elements, grid.num_qp_per_element, 3, 3)
        element_grad = torch.einsum('eqim,eqam->eai', stress, grid.shape_grads)

        gradient = torch.zeros_like(displacements)
        gradient = gradient.index_add(0, grid.elements.reshape(-1),
                                      element_grad.reshape(-1, 3))
        return gradient + self.masses.unsqueeze(-1) * self.gravity

    def potential_energy(self, displacements):
        r"""Total potential energy (elastic plus gravity).

        Args:
            displacements (torch.Tensor): Nodal displacements, of shape
                :math:`(\text{num_nodes}, 3)`.

        Returns:
            torch.Tensor: Scalar energy.
        """
        defo_grads = self.grid.defo_grads(displacements).reshape(-1, 3, 3)
        elastic = neohookean_energy(defo_grads, self.mus, self.lams, self.qp_volumes,
                                    self.reparameterize_lame).sum()
        positions = self.grid.nodes + displacements
        gravity = (self.masses * (positions * self.gravity).sum(-1)).sum()
        return elastic + gravity

    def _tangent_triplets(self, displacements):
        """Element stiffness blocks and their global (row, col) indices."""
        grid = self.grid
        defo_grads = grid.defo_grads(displacements).reshape(-1, 3, 3)
        blocks = neohookean_hessian(defo_grads, self.mus, self.lams, self.qp_volumes,
                                    self.reparameterize_lame)
        blocks = blocks.reshape(grid.num_elements, grid.num_qp_per_element, 3, 3, 3, 3)
        element_stiffness = torch.einsum('eqimjn,eqam,eqbn->eaibj', blocks, grid.shape_grads,
                                        grid.shape_grads)
        element_stiffness = element_stiffness.reshape(grid.num_elements, 24, 24)

        element_dofs = (3 * grid.elements.unsqueeze(-1)
                        + torch.arange(3, device=grid.device)).reshape(grid.num_elements, 24)
        rows = element_dofs.unsqueeze(-1).expand(-1, 24, 24)
        cols = element_dofs.unsqueeze(-2).expand(-1, 24, 24)
        return rows.reshape(-1), cols.reshape(-1), element_stiffness.reshape(-1)

    def _solve_free(self, displacements, residual):
        """Solve the Dirichlet-reduced tangent system for the free degrees of freedom."""
        grid = self.grid
        num_dofs = 3 * grid.num_nodes
        rows, cols, vals = self._tangent_triplets(displacements)

        mass_diag = self.masses.repeat_interleave(3)
        dt2 = self.timestep * self.timestep
        diag_idx = torch.arange(num_dofs, device=grid.device)
        rows = torch.cat([rows, diag_idx])
        cols = torch.cat([cols, diag_idx])
        vals = torch.cat([dt2 * vals, mass_diag])

        free = self.free_dofs
        remap = torch.full((num_dofs,), -1, device=grid.device, dtype=torch.long)
        remap[free] = torch.arange(free.numel(), device=grid.device)
        keep = (remap[rows] >= 0) & (remap[cols] >= 0)
        rows, cols, vals = remap[rows[keep]], remap[cols[keep]], vals[keep]

        rhs = -residual.reshape(-1)[free]
        num_free = free.numel()

        solver = self.linear_solver
        if solver == 'auto':
            solver = 'dense' if num_free <= self.dense_threshold else 'cg'

        if solver == 'dense':
            dense = torch.zeros(num_free, num_free, device=grid.device, dtype=grid.dtype)
            dense.index_put_((rows, cols), vals, accumulate=True)
            return torch.linalg.solve(dense, rhs)

        if solver == 'cg':
            return self._solve_cg(rows, cols, vals, rhs, num_free)

        if solver == 'splu':
            # A CPU sparse direct solve. Exact, but it refactorizes from scratch every Newton
            # iteration, which dominates the run time past a few thousand degrees of freedom.
            from scipy.sparse import coo_matrix
            from scipy.sparse.linalg import splu
            matrix = coo_matrix((vals.detach().cpu().numpy(),
                                 (rows.detach().cpu().numpy(), cols.detach().cpu().numpy())),
                                shape=(num_free, num_free)).tocsc()
            solution = splu(matrix).solve(rhs.detach().cpu().numpy())
            return torch.as_tensor(solution, device=grid.device, dtype=grid.dtype)

        raise ValueError(f'unknown linear_solver {self.linear_solver!r}')

    def _solve_cg(self, rows, cols, vals, rhs, num_free):
        r"""Jacobi-preconditioned conjugate gradients on the assembled tangent.

        The tangent is :math:`M + \Delta t^2 K` with a positive-definite mass term, so it is SPD for
        the deformations reached here and CG is applicable. Unlike the direct path this stays on the
        GPU, which is what makes the reference-resolution beam tractable. The caller checks the
        Newton residual afterwards, so an inexact solve shows up as extra Newton iterations rather
        than as a silently wrong answer.
        """
        indices = torch.stack([rows, cols])
        matrix = torch.sparse_coo_tensor(indices, vals, (num_free, num_free)).coalesce()
        diagonal = torch.zeros(num_free, device=rhs.device, dtype=rhs.dtype)
        matrix_indices = matrix.indices()
        is_diagonal = matrix_indices[0] == matrix_indices[1]
        diagonal = diagonal.index_add(0, matrix_indices[0][is_diagonal],
                                      matrix.values()[is_diagonal])
        inverse_diagonal = torch.where(diagonal.abs() > 0, 1.0 / diagonal,
                                       torch.ones_like(diagonal))
        matrix = matrix.to_sparse_csr()

        solution = torch.zeros_like(rhs)
        residual = rhs.clone()
        preconditioned = inverse_diagonal * residual
        direction = preconditioned.clone()
        rz = residual @ preconditioned
        target = self.cg_tolerance * float(rhs.norm())

        for _ in range(self.cg_max_iterations):
            if float(residual.norm()) <= target:
                break
            matrix_direction = matrix @ direction
            denominator = direction @ matrix_direction
            if float(denominator) <= 0.0:
                # Loss of definiteness: fall back rather than take a meaningless step.
                self.linear_solver = 'splu'
                return self._solve_free_from_triplets(rows, cols, vals, rhs, num_free)
            step = rz / denominator
            solution = solution + step * direction
            residual = residual - step * matrix_direction
            preconditioned = inverse_diagonal * residual
            rz_next = residual @ preconditioned
            direction = preconditioned + (rz_next / rz) * direction
            rz = rz_next

        return solution

    def _solve_free_from_triplets(self, rows, cols, vals, rhs, num_free):
        """CPU sparse direct solve of an already-assembled triplet system."""
        from scipy.sparse import coo_matrix
        from scipy.sparse.linalg import splu
        matrix = coo_matrix((vals.detach().cpu().numpy(),
                             (rows.detach().cpu().numpy(), cols.detach().cpu().numpy())),
                            shape=(num_free, num_free)).tocsc()
        solution = splu(matrix).solve(rhs.detach().cpu().numpy())
        return torch.as_tensor(solution, device=rhs.device, dtype=rhs.dtype)

    def step(self, displacements, velocities, num_newton_steps=20, tolerance=1e-9, verbose=False):
        r"""One backward-Euler step.

        Args:
            displacements (torch.Tensor): Displacements at time :math:`t`, of shape
                :math:`(\text{num_nodes}, 3)`.
            velocities (torch.Tensor): Velocities at time :math:`t`, of shape
                :math:`(\text{num_nodes}, 3)`.
            num_newton_steps (int, optional): Maximum Newton iterations. Default: 20.
            tolerance (float, optional): Relative residual norm at which to stop. Default: 1e-9.
            verbose (bool, optional): Print per-iteration residual norms. Default: False.

        Returns:
            (torch.Tensor, torch.Tensor): displacements and velocities at time
            :math:`t + \Delta t`, each of shape :math:`(\text{num_nodes}, 3)`.
        """
        dt = self.timestep
        trial = displacements + dt * velocities
        if self.pinned_nodes is not None and self.pinned_nodes.numel() > 0:
            trial = trial.clone()
            trial[self.pinned_nodes] = displacements[self.pinned_nodes]

        mass = self.masses.unsqueeze(-1)
        reference = None
        for iteration in range(num_newton_steps):
            inertia = mass * (trial - displacements - dt * velocities)
            residual = inertia + dt * dt * self.potential_gradient(trial)
            norm = float(residual.reshape(-1)[self.free_dofs].norm())
            if reference is None:
                reference = max(norm, 1e-30)
            if verbose:
                print(f'    newton {iteration}: |g| = {norm:.6e}')
            if norm / reference < tolerance:
                break
            update = self._solve_free(trial, residual)
            flat = torch.zeros(3 * self.grid.num_nodes, device=trial.device, dtype=trial.dtype)
            flat[self.free_dofs] = update
            trial = trial + flat.reshape(-1, 3)

        new_velocities = (trial - displacements) / dt
        return trial, new_velocities

    def rollout(self, num_frames, num_newton_steps=20, tolerance=1e-9, verbose=False):
        r"""Simulate from rest for ``num_frames`` steps.

        Args:
            num_frames (int): Number of steps.
            num_newton_steps (int, optional): Maximum Newton iterations per step. Default: 20.
            tolerance (float, optional): Newton stopping tolerance. Default: 1e-9.
            verbose (bool, optional): Print progress. Default: False.

        Returns:
            (torch.Tensor, torch.Tensor): positions and velocities for every frame including the
            rest frame, each of shape :math:`(\text{num_frames} + 1, \text{num_nodes}, 3)`.
        """
        displacements = torch.zeros_like(self.grid.nodes)
        velocities = torch.zeros_like(self.grid.nodes)
        positions = [self.grid.nodes.clone()]
        all_velocities = [velocities.clone()]

        for frame in range(num_frames):
            if verbose:
                print(f'  frame {frame + 1}/{num_frames}')
            displacements, velocities = self.step(displacements, velocities, num_newton_steps,
                                                  tolerance, verbose)
            positions.append(self.grid.nodes + displacements)
            all_velocities.append(velocities.clone())

        return torch.stack(positions, dim=0), torch.stack(all_velocities, dim=0)
