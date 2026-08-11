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

r"""Render trajectories to video, for eyeballing what the error metrics summarize.

A per-vertex L2 number says how far off a rollout is; it does not say whether the failure is a
lagging tip, a wrong bending mode, or a slow drift into a different equilibrium. Those are obvious
on sight and nearly invisible in a scalar, so this module exists to put the reduced rollout and the
full-order trajectory in the same frame.

Deliberately matplotlib-only: no renderer setup, works headless, and the output is a plain mp4.
"""

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.animation import FFMpegWriter, PillowWriter  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

__all__ = [
    'render_trajectory',
]

_SURFACE_COLOR = '#4c78c8'
_GHOST_COLOR = '#bab0ac'


def _as_numpy(tensor):
    return tensor.detach().cpu().numpy() if torch.is_tensor(tensor) else np.asarray(tensor)


def _axis_limits(trajectories, margin=0.05):
    stacked = np.concatenate([t.reshape(-1, 3) for t in trajectories], axis=0)
    lower, upper = stacked.min(axis=0), stacked.max(axis=0)
    span = np.maximum(upper - lower, 1e-6)
    return lower - margin * span, upper + margin * span


def _permutation_for(up_axis):
    r"""Column order that puts ``up_axis`` on matplotlib's vertical (z) axis.

    Simplicits scenes take gravity along :math:`-y`, but a 3D matplotlib axes draws its own z
    vertically, so plotting world coordinates directly makes a falling beam appear to slide into the
    screen. Reordering the columns is what makes the motion read as falling.
    """
    remaining = [axis for axis in range(3) if axis != up_axis]
    return [remaining[0], remaining[1], up_axis]


def render_trajectory(path, positions, faces, reference=None, title='', fps=15, dpi=130,
                      elev=8, azim=-88, limits=None, dt=None, up_axis=1, labels='xyz'):
    r"""Write a trajectory to an mp4 (or gif, if ffmpeg is unavailable).

    Args:
        path (str): Output path. The suffix is replaced with ``.gif`` if ffmpeg is missing.
        positions (torch.Tensor): Positions per frame, of shape
            :math:`(\text{num_frames}, \text{num_nodes}, 3)`.
        faces (torch.Tensor): Surface triangles, of shape :math:`(\text{num_triangles}, 3)`.
        reference (torch.Tensor, optional): Full-order positions to draw as a translucent ghost and
            to measure against, same shape as ``positions``. Default: None.
        title (str, optional): Title prefix. Default: ''.
        fps (int, optional): Frames per second. Default: 15.
        dpi (int, optional): Output resolution. Default: 130.
        elev (float, optional): Camera elevation in degrees. Default: 8.
        azim (float, optional): Camera azimuth in degrees. Default: -88, i.e. nearly side-on, since a
            cantilever's motion is planar.
        limits (tuple, optional): ``(lower, upper)`` bounds in *world* coordinates, so several
            videos can share one frame of reference. Default: None (fit to the data).
        dt (float, optional): Time step, for a clock in the title. Default: None.
        up_axis (int, optional): World axis to draw vertically. Default: 1 (the Simplicits gravity
            axis).
        labels (str, optional): World axis names, for the axis labels. Default: ``'xyz'``.

    Returns:
        str: The path actually written.
    """
    positions = _as_numpy(positions)
    faces = _as_numpy(faces)
    reference = None if reference is None else _as_numpy(reference)

    if limits is None:
        sources = [positions] if reference is None else [positions, reference]
        limits = _axis_limits(sources)

    order = _permutation_for(up_axis)
    positions = positions[..., order]
    if reference is not None:
        reference = reference[..., order]
    lower, upper = limits[0][order], limits[1][order]

    figure = plt.figure(figsize=(9, 5.0))
    axes = figure.add_subplot(111, projection='3d')
    figure.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.88)

    surface = Poly3DCollection(positions[0][faces], facecolor=_SURFACE_COLOR, edgecolor='#1b3a63',
                               linewidths=0.25, alpha=0.97)
    axes.add_collection3d(surface)
    ghost = None
    if reference is not None:
        ghost = Poly3DCollection(reference[0][faces], facecolor=_GHOST_COLOR, edgecolor='none',
                                 alpha=0.30)
        axes.add_collection3d(ghost)

    axes.set_xlim(lower[0], upper[0])
    axes.set_ylim(lower[1], upper[1])
    axes.set_zlim(lower[2], upper[2])
    axes.set_box_aspect(tuple(np.maximum(upper - lower, 1e-6)), zoom=0.92)
    axes.view_init(elev=elev, azim=azim)
    axes.set_xlabel(labels[order[0]])
    axes.set_zlabel(labels[order[2]])
    axes.set_yticks([])
    for pane in (axes.xaxis, axes.yaxis, axes.zaxis):
        pane.pane.set_alpha(0.04)
    text = figure.text(0.5, 0.975, title, ha='center', va='top', fontsize=11)

    def update(frame):
        surface.set_verts(positions[frame][faces])
        label = title
        if dt is not None:
            label = f'{label}   t = {frame * dt:5.2f} s'
        if reference is not None:
            ghost.set_verts(reference[frame][faces])
            error = np.linalg.norm(positions[frame] - reference[frame], axis=-1)
            label = f'{label}   error mean {error.mean():.3f} m  max {error.max():.3f} m'
        text.set_text(label)
        return surface, text

    try:
        writer = FFMpegWriter(fps=fps, bitrate=2400)
        output = path
    except Exception:
        writer = PillowWriter(fps=fps)
        output = str(path).rsplit('.', 1)[0] + '.gif'

    with writer.saving(figure, output, dpi):
        for frame in range(positions.shape[0]):
            update(frame)
            writer.grab_frame()
    plt.close(figure)
    return output
