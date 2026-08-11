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

r"""Generate full-order cantilever-beam trajectories.

The scenario is the repo's own beam regression case, read off
``tests/python/kaolin/physics/simplicits/regression_test_data/``: a
:math:`1 \times 0.25 \times 0.25` beam occupying :math:`[0,1] \times [0.75,1] \times [0.75,1]`,
pinned where :math:`x \geq 0.98`, under gravity :math:`(0, 9.8, 0)` with
:math:`\Delta t = 0.05`, :math:`E = 10^5`, :math:`\nu = 0.45`, :math:`\rho = 500`. At the reference
resolution the grid is :math:`20^3` elements, i.e. the 9261 = :math:`21^3` nodes that
``wpfem_vertex_deformations_beam.pth`` stores.

Unlike that reference -- which holds only frames 0, 1 and 100 -- this writes *every* frame plus the
controls needed to replay the scenario in the reduced model, which is what trajectory training
needs.

Run as::

    python -m kaolin.experimental.simplicits_diffsim.data_gen.gen_fom_beam --coarse
    python -m kaolin.experimental.simplicits_diffsim.data_gen.gen_fom_beam --validate
"""

import argparse
import os
import time

import torch

from .fem_hex import FullOrderNeohookeanSolver, HexGrid

__all__ = [
    'BEAM_SCENARIO',
    'generate_trajectory',
    'load_trajectory',
    'validate_against_reference',
]

BEAM_SCENARIO = {
    'bounds_min': (0.0, 0.75, 0.75),
    'bounds_max': (1.0, 1.0, 1.0),
    'resolution': (20, 20, 20),
    'youngs_modulus': 1e5,
    'poisson_ratio': 0.45,
    'density': 500.0,
    'gravity': (0.0, 9.8, 0.0),
    'timestep': 0.05,
    'pin_axis': 0,
    'pin_threshold': 0.98,
    'reparameterize_lame': True,
}

COARSE_RESOLUTION = (10, 3, 3)

_REFERENCE_PATH = os.path.join('tests', 'python', 'kaolin', 'physics', 'simplicits',
                               'regression_test_data', 'wpfem_vertex_deformations_beam.pth')


def generate_trajectory(num_frames=100, resolution=None, device='cuda', dtype=torch.float64,
                        num_newton_steps=20, tolerance=1e-9, verbose=True, scenario=None,
                        linear_solver='auto'):
    r"""Simulate the beam and return the trajectory together with its replayable controls.

    Args:
        num_frames (int, optional): Number of time steps. Default: 100.
        resolution (sequence of int, optional): Elements per axis. Default: the scenario's.
        device (str or torch.device, optional): Device. Default: ``'cuda'``.
        dtype (torch.dtype, optional): Precision. Default: ``torch.float64``.
        num_newton_steps (int, optional): Maximum Newton iterations per step. Default: 20.
        tolerance (float, optional): Newton stopping tolerance. Default: 1e-9.
        verbose (bool, optional): Print progress. Default: True.
        scenario (dict, optional): Overrides for :data:`BEAM_SCENARIO`. Default: None.
        linear_solver (str, optional): ``'auto'``, ``'dense'``, ``'cg'`` or ``'splu'``; see
            :class:`~.fem_hex.FullOrderNeohookeanSolver`. Default: ``'auto'``.

    Returns:
        dict: With keys ``rest_positions``, ``positions``, ``velocities``, ``faces``,
        ``quadrature_points``, ``nodal_volumes``, ``total_volume``, ``controls`` and ``metadata``.
    """
    config = dict(BEAM_SCENARIO)
    if scenario is not None:
        config.update(scenario)
    if resolution is not None:
        config['resolution'] = tuple(int(r) for r in resolution)

    grid = HexGrid(config['bounds_min'], config['bounds_max'], config['resolution'],
                   device=device, dtype=dtype)
    pinned = torch.nonzero(grid.nodes[:, config['pin_axis']] >= config['pin_threshold'],
                           as_tuple=False).squeeze(1)
    if pinned.numel() == 0:
        raise ValueError('the pin predicate selected no nodes; check pin_axis/pin_threshold')

    solver = FullOrderNeohookeanSolver(grid, config['youngs_modulus'], config['poisson_ratio'],
                                      config['density'], gravity=config['gravity'],
                                      timestep=config['timestep'], pinned_nodes=pinned,
                                      reparameterize_lame=config['reparameterize_lame'],
                                      linear_solver=linear_solver)

    if verbose:
        print(f"grid {config['resolution']}: {grid.num_nodes} nodes, {grid.num_elements} elements, "
              f'{pinned.numel()} pinned, {solver.free_dofs.numel()} free dofs')
    start = time.time()
    positions, velocities = solver.rollout(num_frames, num_newton_steps, tolerance,
                                           verbose=False)
    elapsed = time.time() - start
    if verbose:
        drift = (positions[-1] - grid.nodes).norm(dim=-1).max()
        print(f'{num_frames} frames in {elapsed:.1f}s; max tip displacement {float(drift):.4f} m')

    return {
        'rest_positions': grid.nodes.cpu(),
        'positions': positions.cpu(),
        'velocities': velocities.cpu(),
        'faces': grid.surface_faces().cpu(),
        'quadrature_points': grid.quadrature_points().cpu(),
        'nodal_volumes': grid.nodal_volumes().cpu(),
        'total_volume': float(grid.total_volume()),
        'controls': {
            **{k: v for k, v in config.items()},
            'pinned_nodes': pinned.cpu(),
            'pinned_positions': grid.nodes[pinned].cpu(),
            'integrator': 'backward_euler',
            'num_newton_steps': num_newton_steps,
            'newton_tolerance': tolerance,
        },
        'metadata': {
            'num_frames': num_frames,
            'num_nodes': grid.num_nodes,
            'num_elements': grid.num_elements,
            'element_type': 'Q1_hex_2x2x2_gauss',
            'boundary_conditions': 'hard_dirichlet',
            'dtype': str(dtype),
            'wall_time_seconds': elapsed,
        },
    }


def load_trajectory(path, device=None, dtype=None):
    r"""Load a trajectory written by :func:`generate_trajectory`.

    Args:
        path (str): File path.
        device (torch.device, optional): Device to move tensors to. Default: None (leave on CPU).
        dtype (torch.dtype, optional): Cast floating-point tensors. Default: None (leave as saved).

    Returns:
        dict: The trajectory dict, with tensors moved and cast as requested.
    """
    data = torch.load(path, weights_only=False)

    def convert(value):
        if not torch.is_tensor(value):
            return value
        if device is not None:
            value = value.to(device)
        if dtype is not None and value.is_floating_point():
            value = value.to(dtype)
        return value

    for key, value in list(data.items()):
        if isinstance(value, dict):
            data[key] = {k: convert(v) for k, v in value.items()}
        else:
            data[key] = convert(value)
    return data


def validate_against_reference(reference_path=None, device='cuda', dtype=torch.float64,
                               tolerance=1e-2, linear_solver='auto', newton_tolerance=1e-8):
    r"""Check the generator reproduces the repo's stored 3-frame beam reference.

    Compares frames 0, 1 and 100 by squared chamfer distance -- the same metric and tolerance
    ``tests/python/kaolin/physics/simplicits/test_simplicits_vs_fem.py`` uses -- so the comparison
    does not depend on node ordering.

    Args:
        reference_path (str, optional): Path to ``wpfem_vertex_deformations_beam.pth``. Default:
            the in-repo location, resolved relative to the current directory.
        device (str or torch.device, optional): Device. Default: ``'cuda'``.
        dtype (torch.dtype, optional): Precision. Default: ``torch.float64``.
        tolerance (float, optional): Chamfer tolerance, matching the existing test. Default: 1e-2.
        linear_solver (str, optional): Linear solver for the full-order steps. Default: ``'auto'``.
        newton_tolerance (float, optional): Newton stopping tolerance. Default: 1e-8.

    Returns:
        dict: Chamfer distances keyed by ``'v0'``, ``'v1'`` and ``'v_end'``.
    """
    from kaolin.metrics.pointcloud import chamfer_distance

    path = reference_path or _REFERENCE_PATH
    reference = torch.load(path, weights_only=False)
    trajectory = generate_trajectory(num_frames=100, device=device, dtype=dtype, verbose=True,
                                     linear_solver=linear_solver, tolerance=newton_tolerance)
    positions = trajectory['positions'].to(device=device, dtype=torch.float32)

    results = {}
    for key, frame in (('v0', 0), ('v1', 1), ('v_end', 100)):
        expected = reference[key].to(device=device, dtype=torch.float32)
        distance = chamfer_distance(expected.unsqueeze(0), positions[frame].unsqueeze(0),
                                    w1=1.0, w2=1.0, squared=True)
        results[key] = float(distance[0])
        limit = tolerance if key == 'v_end' else tolerance * tolerance + 1e-5
        status = 'ok' if results[key] < limit else 'FAIL'
        print(f'  {key:6s} chamfer {results[key]:.3e}  (limit {limit:.3e})  {status}')
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--coarse', action='store_true',
                        help=f'use a {COARSE_RESOLUTION} grid so training iterates in seconds')
    parser.add_argument('--resolution', type=int, nargs=3, default=None,
                        help='elements per axis, overriding --coarse')
    parser.add_argument('--frames', type=int, default=100, help='number of time steps')
    parser.add_argument('--out', type=str, default=None, help='output .pth path')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--float32', action='store_true', help='generate in single precision')
    parser.add_argument('--validate', action='store_true',
                        help="only check the generator against the repo's 3-frame reference")
    parser.add_argument('--linear-solver', type=str, default='auto',
                        choices=['auto', 'dense', 'cg', 'splu'],
                        help='linear solver for the full-order Newton steps')
    parser.add_argument('--newton-tol', type=float, default=1e-9,
                        help='relative Newton residual at which a step is considered converged')
    args = parser.parse_args()

    dtype = torch.float32 if args.float32 else torch.float64
    if args.validate:
        validate_against_reference(device=args.device, dtype=dtype,
                                   linear_solver=args.linear_solver,
                                   newton_tolerance=args.newton_tol)
        return

    resolution = args.resolution
    if resolution is None and args.coarse:
        resolution = COARSE_RESOLUTION
    trajectory = generate_trajectory(num_frames=args.frames, resolution=resolution,
                                     device=args.device, dtype=dtype,
                                     linear_solver=args.linear_solver,
                                     tolerance=args.newton_tol)

    out = args.out
    if out is None:
        tag = 'x'.join(str(r) for r in trajectory['controls']['resolution'])
        out = os.path.join('data', f'fom_beam_{tag}_{args.frames}frames.pth')
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    torch.save(trajectory, out)
    print(f"wrote {out} ({trajectory['metadata']['num_nodes']} nodes, {args.frames} frames)")


if __name__ == '__main__':
    main()
