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

r"""Full-order trajectories, sliced into training windows.

A window is a short stretch of a full-order trajectory the reduced model is asked to reproduce.
Windows that do not start at rest need an initial reduced state, and there is no ground truth for
one -- the full-order state lives in a much larger space. The honest choice, and the one used here,
is the best-fit projection of the full-order state onto the current basis
(:func:`~.projection.best_fit_reduced_coords`), which is the closest the reduced model could
possibly start. Because the basis moves during training, that projection is recomputed each time a
window is used.

Train/eval splits are by *frame range* rather than at random: a random split would let the model
interpolate between neighbouring frames it has already seen, which says nothing about whether the
learned dynamics extrapolate.

A dataset holds several trajectories of *one* object under different loads. That is the only way to
scale training data for a Simplicits field: :math:`W_\theta` maps a point of this object's bounding
box to this object's handle weights, so it is per-object by construction and more data means more
scenarios and more frames, not more geometries.

Which controls may vary across those scenarios is enforced rather than assumed. ``gravity`` may
differ; the time step, material and pin predicate may not, because the trainer bakes them into the
quadrature weights and the reduced mass matrix once. Without that check a dataset of
different-material scenarios would train every one of them against trajectory 0's material and
report a plausible-looking loss.
"""

import torch

from .projection import best_fit_reduced_coords

__all__ = [
    'TrajectoryWindow',
    'TrajectoryDataset',
]

# Controls every trajectory in a dataset must agree on, because the trainer resolves them once into
# quadrature volumes, Lame parameters, masses and pin indices that are shared across all scenarios.
_SHARED_CONTROL_KEYS = ('timestep', 'youngs_modulus', 'poisson_ratio', 'density', 'pin_axis',
                        'pin_threshold')

# Controls that are allowed to differ per scenario. Each one costs only a force rebuild, never a
# rebuild of B, dF/dz or B^T M B, which is what makes many scenarios per step affordable.
_SCENARIO_CONTROL_KEYS = ('gravity',)


class TrajectoryWindow:
    r"""One training window: a slice of a full-order trajectory plus where it starts.

    Args:
        trajectory_index (int): Which trajectory in the dataset this came from.
        start_frame (int): Index of the frame the window starts *from*; targets begin at
            ``start_frame + 1``.
        target_positions (torch.Tensor): Full-order positions to match, of shape
            :math:`(\text{horizon}, \text{num_nodes}, 3)`.
        initial_displacements (torch.Tensor): Full-order displacements at ``start_frame``, of shape
            :math:`(\text{num_nodes}, 3)`.
        initial_velocities (torch.Tensor): Full-order velocities at ``start_frame``, of shape
            :math:`(\text{num_nodes}, 3)`.
        starts_at_rest (bool): True when ``start_frame == 0``, in which case the reduced initial
            state is exactly zero and needs no projection.
    """

    def __init__(self, trajectory_index, start_frame, target_positions, initial_displacements,
                 initial_velocities, starts_at_rest):
        self.trajectory_index = trajectory_index
        self.start_frame = start_frame
        self.target_positions = target_positions
        self.initial_displacements = initial_displacements
        self.initial_velocities = initial_velocities
        self.starts_at_rest = starts_at_rest

    @property
    def horizon(self):
        r"""int: Number of frames the reduced model must predict."""
        return self.target_positions.shape[0]

    def initial_reduced_state(self, decode_lbs, node_masses=None, ridge=1e-8):
        r"""Reduced coordinates and velocity to start the rollout from.

        Args:
            decode_lbs (torch.Tensor): Skinning matrix at the full-order nodes, of shape
                :math:`(3 \text{num_nodes}, 12 \text{num_handles})`.
            node_masses (torch.Tensor, optional): Per-node masses defining the projection norm,
                of shape :math:`(\text{num_nodes},)`. Default: None.
            ridge (float, optional): Tikhonov regularization for the projection. Default: 1e-8.

        Returns:
            (torch.Tensor, torch.Tensor): reduced coordinates and reduced velocity, each of shape
            :math:`(12 \text{num_handles},)`.
        """
        if self.starts_at_rest:
            zeros = decode_lbs.new_zeros(decode_lbs.shape[1])
            return zeros, zeros
        coords = best_fit_reduced_coords(decode_lbs, self.initial_displacements, node_masses, ridge)
        velocity = best_fit_reduced_coords(decode_lbs, self.initial_velocities, node_masses, ridge)
        return coords, velocity


class TrajectoryDataset:
    r"""A collection of full-order trajectories sharing one rest configuration.

    Args:
        trajectories (list of dict): Trajectories as written by
            :func:`~.data_gen.gen_fom_beam.generate_trajectory`.
        device (torch.device, optional): Device to hold the tensors on. Default: None.
        dtype (torch.dtype, optional): Precision. Default: ``torch.float64``.
        train_fraction (float, optional): Fraction of each trajectory's frames used for training;
            the remainder is held out for evaluation. Default: 0.7.
    """

    def __init__(self, trajectories, device=None, dtype=torch.float64, train_fraction=0.7):
        if not trajectories:
            raise ValueError('no trajectories given')
        self.device = device
        self.dtype = dtype
        self.train_fraction = train_fraction

        self.trajectories = []
        for data in trajectories:
            self.trajectories.append({
                'rest_positions': data['rest_positions'].to(device=device, dtype=dtype),
                'positions': data['positions'].to(device=device, dtype=dtype),
                'velocities': data['velocities'].to(device=device, dtype=dtype),
                'nodal_volumes': data['nodal_volumes'].to(device=device, dtype=dtype),
                'total_volume': float(data['total_volume']),
                'controls': data['controls'],
                'metadata': data.get('metadata', {}),
            })

        first = self.trajectories[0]
        self.rest_positions = first['rest_positions']
        self.nodal_volumes = first['nodal_volumes']
        self.total_volume = first['total_volume']
        self.controls = first['controls']
        self.timestep = float(self.controls['timestep'])
        for index, other in enumerate(self.trajectories[1:], start=1):
            if not torch.equal(other['rest_positions'], self.rest_positions):
                raise ValueError('all trajectories must share one rest configuration')
            for key in _SHARED_CONTROL_KEYS:
                mine, theirs = self.controls.get(key), other['controls'].get(key)
                if mine != theirs:
                    raise ValueError(
                        f'trajectory {index} has {key}={theirs!r} but trajectory 0 has {mine!r}; '
                        f'{_SHARED_CONTROL_KEYS} must match across a dataset because the trainer '
                        f'resolves them once. Varying {key} needs a separate dataset, or a trainer '
                        f'that rebuilds the reduced mass matrix per scenario.')

    @classmethod
    def from_files(cls, paths, device=None, dtype=torch.float64, train_fraction=0.7):
        r"""Load trajectories from disk.

        Args:
            paths (str or list of str): Paths written by
                :func:`~.data_gen.gen_fom_beam.generate_trajectory`.
            device (torch.device, optional): Device. Default: None.
            dtype (torch.dtype, optional): Precision. Default: ``torch.float64``.
            train_fraction (float, optional): Train/eval frame split. Default: 0.7.

        Returns:
            TrajectoryDataset: The loaded dataset.
        """
        from .data_gen.gen_fom_beam import load_trajectory

        if isinstance(paths, str):
            paths = [paths]
        return cls([load_trajectory(p) for p in paths], device=device, dtype=dtype,
                   train_fraction=train_fraction)

    @property
    def num_trajectories(self):
        r"""int: Number of scenarios in the dataset."""
        return len(self.trajectories)

    def controls_at(self, trajectory_index=0):
        r"""Replay controls for one scenario.

        Args:
            trajectory_index (int, optional): Which trajectory. Default: 0.

        Returns:
            dict: The trajectory's ``controls``.
        """
        return self.trajectories[trajectory_index]['controls']

    def gravity_at(self, trajectory_index=0):
        r"""Gravity vector of one scenario, as a tensor on the dataset's device and dtype.

        Args:
            trajectory_index (int, optional): Which trajectory. Default: 0.

        Returns:
            torch.Tensor: Acceleration vector, of shape :math:`(3,)` (in :math:`m/s^2`).
        """
        return torch.as_tensor(self.controls_at(trajectory_index)['gravity'],
                               device=self.rest_positions.device, dtype=self.rest_positions.dtype)

    def scenario_summary(self):
        r"""One-line description per scenario, for logging which loads are in the pool.

        Returns:
            list of str: Descriptions, one per trajectory.
        """
        lines = []
        for index in range(self.num_trajectories):
            gravity = self.gravity_at(index)
            direction = gravity / gravity.norm().clamp_min(1e-12)
            lines.append(f'scenario {index:2d}: |g| {float(gravity.norm()):5.2f} '
                         f"dir ({direction[0]:+.2f}, {direction[1]:+.2f}, {direction[2]:+.2f})  "
                         f'{self.num_frames(index)} frames')
        return lines

    def num_frames(self, trajectory_index=0):
        r"""Number of simulated frames, excluding the rest frame.

        Args:
            trajectory_index (int, optional): Which trajectory. Default: 0.

        Returns:
            int: Frame count.
        """
        return self.trajectories[trajectory_index]['positions'].shape[0] - 1

    def split_frame(self, trajectory_index=0):
        r"""Last frame index belonging to the training range.

        Args:
            trajectory_index (int, optional): Which trajectory. Default: 0.

        Returns:
            int: Frame index.
        """
        return int(self.train_fraction * self.num_frames(trajectory_index))

    def window(self, horizon, start_frame=0, trajectory_index=0):
        r"""Build a single window.

        Args:
            horizon (int): Number of frames to predict.
            start_frame (int, optional): Frame the window starts from. Default: 0.
            trajectory_index (int, optional): Which trajectory. Default: 0.

        Returns:
            TrajectoryWindow: The window.
        """
        data = self.trajectories[trajectory_index]
        last = start_frame + horizon
        if last > data['positions'].shape[0] - 1:
            raise ValueError(f'window [{start_frame}, {last}] runs past the trajectory '
                             f"({data['positions'].shape[0] - 1} frames)")
        rest = data['rest_positions']
        return TrajectoryWindow(
            trajectory_index=trajectory_index,
            start_frame=start_frame,
            target_positions=data['positions'][start_frame + 1:last + 1],
            initial_displacements=data['positions'][start_frame] - rest,
            initial_velocities=data['velocities'][start_frame],
            starts_at_rest=(start_frame == 0))

    def train_windows(self, horizon, stride=1, trajectory_index=0):
        r"""All training-range windows of a given horizon.

        Args:
            horizon (int): Frames per window.
            stride (int, optional): Step between window starts. Default: 1.
            trajectory_index (int, optional): Which trajectory. Default: 0.

        Returns:
            list of TrajectoryWindow: The windows.
        """
        last_start = self.split_frame(trajectory_index) - horizon
        starts = range(0, max(last_start, 0) + 1, stride)
        return [self.window(horizon, s, trajectory_index) for s in starts]

    def eval_window(self, horizon=None, trajectory_index=0):
        r"""A held-out window starting at the split point.

        Args:
            horizon (int, optional): Frames to predict. Default: everything after the split.
            trajectory_index (int, optional): Which trajectory. Default: 0.

        Returns:
            TrajectoryWindow: The window.
        """
        split = self.split_frame(trajectory_index)
        available = self.num_frames(trajectory_index) - split
        return self.window(horizon or available, split, trajectory_index)

    def full_window(self, trajectory_index=0):
        r"""The whole trajectory as one window starting from rest.

        Args:
            trajectory_index (int, optional): Which trajectory. Default: 0.

        Returns:
            TrajectoryWindow: The window.
        """
        return self.window(self.num_frames(trajectory_index), 0, trajectory_index)

    def all_train_windows(self, horizon, stride=1):
        r"""Training-range windows from *every* scenario, pooled.

        This is the pool trajectory training should draw from: with one scenario a field can fit
        the single deformation family that trajectory visits, and does so well before it has
        learned a basis for the object.

        Args:
            horizon (int): Frames per window.
            stride (int, optional): Step between window starts. Default: 1.

        Returns:
            list of TrajectoryWindow: Windows from all trajectories, in scenario order.
        """
        pooled = []
        for index in range(self.num_trajectories):
            pooled.extend(self.train_windows(horizon, stride=stride, trajectory_index=index))
        return pooled

    def all_eval_windows(self, horizon=None):
        r"""One held-out window per scenario.

        Args:
            horizon (int, optional): Frames to predict. Default: everything after each split.

        Returns:
            list of TrajectoryWindow: Held-out windows, one per trajectory.
        """
        return [self.eval_window(horizon, index) for index in range(self.num_trajectories)]

    def all_full_windows(self):
        r"""Every scenario as one window starting from rest.

        Returns:
            list of TrajectoryWindow: Full-trajectory windows, one per trajectory.
        """
        return [self.full_window(index) for index in range(self.num_trajectories)]

    def train_snapshots(self):
        r"""Full-order displacements over every scenario's training range, stacked.

        Used by snapshot pretraining and by the POD baseline, both of which want *all* the shapes
        the basis must represent rather than one trajectory's worth.

        Returns:
            torch.Tensor: Displacements, of shape
            :math:`(\sum_s \text{split}_s, \text{num_nodes}, 3)`.
        """
        chunks = []
        for index in range(self.num_trajectories):
            data = self.trajectories[index]
            split = self.split_frame(index)
            chunks.append(data['positions'][1:split + 1] - data['rest_positions'])
        return torch.cat(chunks, dim=0)

    def quadrature_points(self, num_points, mode='random', generator=None, trajectory_index=0):
        r"""Quadrature points and their integration volumes for the reduced model.

        Args:
            num_points (int): Number of points to return. Ignored when ``mode='nodes'``.
            mode (str, optional): ``'random'`` for uniform Monte-Carlo points in the bounding box
                with volumes :math:`V/N`, which is what ``SimplicitsScene`` does and which leaves a
                cubature error floor worth reporting; ``'fom'`` to subsample the full-order
                quadrature points; ``'nodes'`` to use the full-order nodes with their lumped
                volumes, which removes the cubature mismatch entirely and so isolates reduction
                error. Default: ``'random'``.
            generator (torch.Generator, optional): Generator for the sampling. Default: None.
            trajectory_index (int, optional): Which trajectory. Default: 0.

        Returns:
            (torch.Tensor, torch.Tensor): points of shape :math:`(\text{num_points}, 3)` and
            volumes of shape :math:`(\text{num_points},)`.
        """
        data = self.trajectories[trajectory_index]
        rest = data['rest_positions']

        if mode == 'nodes':
            return rest, data['nodal_volumes']

        if mode == 'fom':
            source = data.get('quadrature_points')
            if source is None:
                raise ValueError("this trajectory has no stored quadrature points; use 'random'")
            source = source.to(device=rest.device, dtype=rest.dtype)
            index = torch.randperm(source.shape[0], device=rest.device,
                                   generator=generator)[:num_points]
            points = source[index]
        elif mode == 'random':
            lower, upper = rest.min(dim=0).values, rest.max(dim=0).values
            unit = torch.rand(num_points, 3, device=rest.device, dtype=rest.dtype,
                              generator=generator)
            points = lower + unit * (upper - lower)
        else:
            raise ValueError(f"unknown quadrature mode {mode!r}")

        volumes = torch.full((points.shape[0],), self.total_volume / points.shape[0],
                             device=rest.device, dtype=rest.dtype)
        return points, volumes

    def material_at(self, points):
        r"""Point-wise material parameters, taken from the trajectory's controls.

        Args:
            points (torch.Tensor): Query points, of shape :math:`(\text{num_points}, 3)`.

        Returns:
            (torch.Tensor, torch.Tensor, torch.Tensor): Young's modulus, Poisson's ratio and
            density, each of shape :math:`(\text{num_points},)`.
        """
        count = points.shape[0]
        kwargs = dict(device=points.device, dtype=points.dtype)
        return (torch.full((count,), float(self.controls['youngs_modulus']), **kwargs),
                torch.full((count,), float(self.controls['poisson_ratio']), **kwargs),
                torch.full((count,), float(self.controls['density']), **kwargs))
