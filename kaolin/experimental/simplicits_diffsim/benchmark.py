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

r"""Wall-clock cost of a simulation step, as frames per second and real-time factor.

Speed is the entire reason for model reduction, so a reduced model that matches the full-order
trajectory is only interesting alongside what it cost. Two things this module is careful about,
because getting either wrong inflates a reduced model's apparent advantage:

* **Synchronize.** Both the torch twin and the Warp scene launch asynchronously, so an untimed
  ``cuda.synchronize()`` moves the cost of frame *n* into frame *n+1* and, at the end, out of the
  measurement entirely.
* **Say which implementation.** The twin is a *research* forward path -- dense operators, dense
  ``linalg.solve``, float64 by default -- kept simple so it is differentiable and auditable. The
  production Warp path solves the same reduced system far faster. Quoting the twin's timing as the
  speed of Simplicits would understate the method by a large factor, so
  :func:`time_warp_scene` measures that path too.

Real-time factor is simulated seconds per wall-clock second, :math:`\Delta t \, / \, t_{step}`; above
1 means faster than real time.
"""

import time

import torch

__all__ = [
    'StepTiming',
    'time_full_order',
    'time_reduced',
    'time_warp_scene',
]


class StepTiming:
    r"""Timing of one simulation, as seconds per frame plus derived rates.

    Args:
        label (str): What was measured.
        seconds_per_frame (float): Mean wall-clock seconds per simulated frame.
        timestep (float): Simulated seconds per frame (in :math:`s`).
        num_dofs (int): Degrees of freedom solved for.
        detail (str, optional): Free-form note on the configuration. Default: ''.
    """

    def __init__(self, label, seconds_per_frame, timestep, num_dofs, detail=''):
        self.label = label
        self.seconds_per_frame = seconds_per_frame
        self.timestep = timestep
        self.num_dofs = num_dofs
        self.detail = detail

    @property
    def fps(self):
        r"""float: Simulated frames per wall-clock second."""
        return 1.0 / self.seconds_per_frame

    @property
    def rtf(self):
        r"""float: Real-time factor, simulated seconds per wall-clock second."""
        return self.timestep / self.seconds_per_frame

    def __str__(self):
        return (f'{self.label:<34} {self.num_dofs:>7} dofs  '
                f'{self.seconds_per_frame * 1e3:9.2f} ms/frame  {self.fps:8.1f} fps  '
                f'RTF {self.rtf:7.3f}   {self.detail}')


def _timed(step_fn, num_frames, warmup):
    """Run ``step_fn`` ``warmup + num_frames`` times, timing only the tail, with syncs."""
    for _ in range(warmup):
        step_fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_frames):
        step_fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - start) / num_frames


def time_full_order(solver, num_frames=20, warmup=3, num_newton_steps=20, tolerance=1e-9):
    r"""Time the full-order solver.

    Steps are taken from a *moving* state rather than repeatedly from rest: Newton needs more
    iterations while the beam is accelerating than near equilibrium, so timing only rest-state steps
    would understate the cost.

    Args:
        solver (FullOrderNeohookeanSolver): The solver to time.
        num_frames (int, optional): Timed frames. Default: 20.
        warmup (int, optional): Untimed frames first. Default: 3.
        num_newton_steps (int, optional): Newton cap per step. Default: 20.
        tolerance (float, optional): Newton stopping tolerance. Default: 1e-9.

    Returns:
        StepTiming: The measurement.
    """
    state = {'u': torch.zeros_like(solver.grid.nodes),
             'v': torch.zeros_like(solver.grid.nodes)}

    def step():
        state['u'], state['v'] = solver.step(state['u'], state['v'], num_newton_steps, tolerance)

    seconds = _timed(step, num_frames, warmup)
    return StepTiming(f'full-order FEM ({solver.linear_solver})', seconds, solver.timestep,
                      int(solver.free_dofs.numel()),
                      f'{solver.grid.num_elements} elements, {solver.grid.dtype}')


def time_reduced(model, timestep, num_frames=20, warmup=3, num_newton_steps=4, line_search=True,
                 label='reduced (torch twin)'):
    r"""Time the differentiable twin's forward path, under ``no_grad``.

    Args:
        model (ReducedModel): The reduced system. Assembly of :math:`B` and
            :math:`\partial F/\partial z` is *excluded*: it happens once at bake time, not per frame.
        timestep (float): Time step (in :math:`s`).
        num_frames (int, optional): Timed frames. Default: 20.
        warmup (int, optional): Untimed frames first. Default: 3.
        num_newton_steps (int, optional): Unrolled Newton iterations. Default: 4.
        line_search (bool, optional): Whether to line-search. Default: True.
        label (str, optional): Label for the result. Default: ``'reduced (torch twin)'``.

    Returns:
        StepTiming: The measurement.
    """
    from .step import newton_step_unrolled

    state = {'z': model.zeros(), 'v': model.zeros()}

    @torch.no_grad()
    def step():
        nxt = newton_step_unrolled(model, state['z'], state['v'], timestep,
                                   num_newton_steps=num_newton_steps, line_search=line_search)
        state['v'] = (nxt - state['z']) / timestep
        state['z'] = nxt

    seconds = _timed(step, num_frames, warmup)
    return StepTiming(label, seconds, timestep, int(model.num_reduced_dofs),
                      f'{model.num_samples} qp, {num_newton_steps} newton, '
                      f'{model.lbs.dtype}')


def time_warp_scene(scene, num_frames=20, warmup=3, label='reduced (production Warp)'):
    r"""Time ``SimplicitsScene.run_sim_step``, the production forward path.

    Args:
        scene (kaolin.physics.simplicits.SimplicitsScene): A scene ready to step.
        num_frames (int, optional): Timed frames. Default: 20.
        warmup (int, optional): Untimed frames first. Default: 3.
        label (str, optional): Label for the result. Default: ``'reduced (production Warp)'``.

    Returns:
        StepTiming: The measurement.
    """
    obj = scene.get_object(0)
    num_dofs = 12 * obj.num_handles
    detail = f'{obj.num_qp} qp, {scene.max_newton_steps} newton max, torch.float32'
    seconds = _timed(scene.run_sim_step, num_frames, warmup)
    return StepTiming(label, seconds, scene.timestep, num_dofs, detail)
