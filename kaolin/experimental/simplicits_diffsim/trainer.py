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

r"""Train a skinning field through the reduced simulator.

The outer loop is ordinary: sample a short window of a full-order trajectory, roll the reduced
model out over it, compare, backpropagate into :math:`\theta`. What makes it work at all is that
every operator in that rollout is differentiable, which is the point of this package.

Three things are reported at every evaluation, and the third is the one that keeps the numbers
honest:

* **rollout error** -- per-vertex L2 against the full-order trajectory, on held-out frames.
* **projection error** -- the best any :math:`z` could do in the *current* basis. Rollout error can
  never beat it, so the gap between the two is what training can still recover; the floor itself
  needs more handles, not more steps.
* **the data-free objective** -- what the original Simplicits loss would have said, tracked even
  when its coefficient is zero, so a field that has drifted into something only valid near the
  sampled trajectories is visible rather than inferred.

Run as::

    python -m kaolin.experimental.simplicits_diffsim.trainer --beam --horizon 4
"""

import argparse
import os
import time

import torch

from kaolin.physics.materials.material_utils import to_lame
from kaolin.physics.simplicits.network import SimplicitsMLP

from . import forces
from .dataset import TrajectoryDataset
from .kinematics import dense_lbs_matrix, qr_reparameterization
from .losses import data_free_regularizer, per_vertex_l2_error, trajectory_loss
from .projection import projection_error
from .reduced_model import build_reduced_model
from .rollout import SkinningDecoder, rollout

__all__ = [
    'TrainerConfig',
    'SimInLoopTrainer',
]


class TrainerConfig:
    r"""Configuration for :class:`SimInLoopTrainer`.

    Args:
        num_qp (int, optional): Quadrature points driving the reduced dynamics. Default: 512.
        quadrature_mode (str, optional): See
            :meth:`~.dataset.TrajectoryDataset.quadrature_points`. Default: ``'random'``.
        num_newton_steps (int, optional): Unrolled Newton iterations per step. Default: 4.
        line_search (bool, optional): Line-search each Newton direction. Default: True.
        bdry_penalty (float, optional): Pin stiffness for the reduced model. The full-order solver
            uses hard Dirichlet constraints, so this is an approximation of them and a genuine
            source of mismatch. Default: 10000.0.
        use_qr (bool, optional): Reparameterize the reduced coordinates by the QR factor of
            :math:`B`. The rollout is affine-invariant so the loss value is unchanged, but the
            linear solves are much better conditioned. Default: True.
        hessian_regularizer (float, optional): Added to :math:`H` inside the solve only.
            Default: 0.0.
        pos_coeff (float, optional): Position loss weight. Default: 1.0.
        vel_coeff (float, optional): Velocity loss weight. Default: 0.0.
        data_free_coeff (float, optional): Initial weight on the original Simplicits objective, as
            a regularizer. Default: 0.0.
        data_free_decay (float, optional): Multiplied into ``data_free_coeff`` each step.
            Default: 1.0.
        learning_rate (float, optional): Adam learning rate. Default: 1e-4.
        bptt_window (int, optional): Truncated-BPTT window. Default: None (full BPTT).
        grad_clip (float, optional): Gradient-norm clip; 0 disables. Default: 1.0.
    """

    def __init__(self, num_qp=512, quadrature_mode='random', num_newton_steps=4, line_search=True,
                 bdry_penalty=10000.0, use_qr=True, hessian_regularizer=0.0, pos_coeff=1.0,
                 vel_coeff=0.0, data_free_coeff=0.0, data_free_decay=1.0, learning_rate=1e-4,
                 bptt_window=None, grad_clip=1.0):
        self.num_qp = num_qp
        self.quadrature_mode = quadrature_mode
        self.num_newton_steps = num_newton_steps
        self.line_search = line_search
        self.bdry_penalty = bdry_penalty
        self.use_qr = use_qr
        self.hessian_regularizer = hessian_regularizer
        self.pos_coeff = pos_coeff
        self.vel_coeff = vel_coeff
        self.data_free_coeff = data_free_coeff
        self.data_free_decay = data_free_decay
        self.learning_rate = learning_rate
        self.bptt_window = bptt_window
        self.grad_clip = grad_clip


class SimInLoopTrainer:
    r"""Trains a :class:`~kaolin.physics.simplicits.network.SkinningModule` through the reduced
    simulator.

    Args:
        skinning_mod (kaolin.physics.simplicits.network.SkinningModule): Field to train. Its
            bounding box must cover the object, since
            :meth:`~kaolin.physics.simplicits.network.SkinningModule.compute_skinning_weights`
            normalizes with it.
        dataset (TrajectoryDataset): Full-order trajectories to match.
        config (TrainerConfig, optional): Hyperparameters. Default: defaults of
            :class:`TrainerConfig`.
        generator (torch.Generator, optional): Generator for quadrature sampling. Default: None.
    """

    def __init__(self, skinning_mod, dataset, config=None, generator=None):
        self.skinning_mod = skinning_mod
        self.dataset = dataset
        self.config = config or TrainerConfig()
        self.timestep = dataset.timestep

        points, volumes = dataset.quadrature_points(self.config.num_qp,
                                                    mode=self.config.quadrature_mode,
                                                    generator=generator)
        self.points = points
        self.volumes = volumes
        yms, prs, rhos = dataset.material_at(points)
        self.rhos = rhos
        self.mus, self.lams = to_lame(yms, prs)
        self.masses = rhos * volumes

        controls = dataset.controls
        gravity = torch.as_tensor(controls['gravity'], device=points.device, dtype=points.dtype)
        pin_axis = int(controls['pin_axis'])
        pin_threshold = float(controls['pin_threshold'])
        pinned = torch.nonzero(points[:, pin_axis] >= pin_threshold, as_tuple=False).squeeze(1)
        if pinned.numel() == 0:
            raise ValueError('no quadrature point satisfies the pin predicate; raise num_qp')
        self.pt_forces = [forces.Gravity(gravity, rhos, volumes),
                          forces.Boundary(pinned, points[pinned].clone(),
                                          self.config.bdry_penalty)]
        self.num_pinned = int(pinned.numel())

        self.node_masses = dataset.nodal_volumes * float(controls['density'])
        self.optimizer = torch.optim.Adam(skinning_mod.parameters(), lr=self.config.learning_rate)
        self.current_data_free_coeff = self.config.data_free_coeff
        self.history = []

    def build(self):
        r"""Assemble the reduced model and the full-order-node decoder for the current
        :math:`\theta`.

        Both must be rebuilt after every optimizer step: they hold :math:`B` and
        :math:`\partial F/\partial z` as live functions of the parameters.

        Returns:
            (ReducedModel, SkinningDecoder): the reduced system and the decoder.
        """
        qr_transform = None
        if self.config.use_qr:
            weights = self.skinning_mod.compute_skinning_weights(self.points)
            qr_transform = qr_reparameterization(dense_lbs_matrix(self.points, weights).detach())
        model = build_reduced_model(self.skinning_mod, self.points, self.mus, self.lams,
                                   self.volumes, self.masses, pt_forces=self.pt_forces,
                                   qr_transform=qr_transform)
        decoder = SkinningDecoder(self.skinning_mod, self.dataset.rest_positions, qr_transform)
        return model, decoder

    def predict(self, window, model=None, decoder=None):
        r"""Roll the reduced model out over a window and decode it at the full-order nodes.

        Args:
            window (TrajectoryWindow): The window to predict.
            model (ReducedModel, optional): Prebuilt model. Default: None (build one).
            decoder (SkinningDecoder, optional): Prebuilt decoder. Default: None.

        Returns:
            torch.Tensor: Predicted positions, of shape
            :math:`(\text{horizon}, \text{num_nodes}, 3)`.
        """
        if model is None or decoder is None:
            model, decoder = self.build()
        coords, velocity = window.initial_reduced_state(decoder.lbs, self.node_masses)
        coords_traj, _ = rollout(model, self.timestep, window.horizon,
                                 reduced_coords=coords, reduced_velocity=velocity,
                                 num_newton_steps=self.config.num_newton_steps,
                                 line_search=self.config.line_search,
                                 hessian_regularizer=self.config.hessian_regularizer,
                                 bptt_window=self.config.bptt_window)
        return decoder.trajectory(coords_traj)

    def window_loss(self, window):
        r"""Trajectory loss for one window, plus the optional data-free regularizer.

        Args:
            window (TrajectoryWindow): The window.

        Returns:
            (torch.Tensor, dict): the scalar loss and a dict of logged terms.
        """
        model, decoder = self.build()
        predicted = self.predict(window, model, decoder)
        loss, terms = trajectory_loss(predicted, window.target_positions, self.timestep,
                                      node_weights=self.dataset.nodal_volumes,
                                      pos_coeff=self.config.pos_coeff,
                                      vel_coeff=self.config.vel_coeff)

        if self.current_data_free_coeff > 0.0:
            normalized = self.skinning_mod._offset_scale(self.points)
            yms, prs, rhos = self.dataset.material_at(self.points)
            elastic, ortho = data_free_regularizer(self.skinning_mod, normalized, yms, prs, rhos,
                                                   self.dataset.total_volume,
                                                   num_samples=min(512, self.points.shape[0]))
            loss = loss + self.current_data_free_coeff * (elastic + ortho)
            terms['data_free_elastic'] = float(elastic)
            terms['data_free_ortho'] = float(ortho)

        terms['start_frame'] = window.start_frame
        terms['horizon'] = window.horizon
        return loss, terms

    def train_step(self, window):
        r"""One optimizer step on one window.

        Args:
            window (TrajectoryWindow): The window.

        Returns:
            dict: Logged terms, including ``loss`` and ``grad_norm``.
        """
        self.optimizer.zero_grad(set_to_none=True)
        loss, terms = self.window_loss(window)
        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.skinning_mod.parameters(),
            self.config.grad_clip if self.config.grad_clip > 0 else float('inf'))
        self.optimizer.step()
        self.current_data_free_coeff *= self.config.data_free_decay

        terms['loss'] = float(loss)
        terms['grad_norm'] = float(grad_norm)
        return terms

    @torch.no_grad()
    def evaluate(self, window):
        r"""Rollout error on a window, reported next to the projection floor.

        Args:
            window (TrajectoryWindow): Usually a held-out window.

        Returns:
            dict: ``rollout_mean``, ``rollout_max``, ``projection_mean``, ``projection_max`` (all
            per-vertex distances in :math:`m`) and ``target_max`` for scale.
        """
        model, decoder = self.build()
        predicted = self.predict(window, model, decoder)
        distances = per_vertex_l2_error(predicted, window.target_positions)

        rest = self.dataset.rest_positions
        displacements = window.target_positions - rest
        _, _, reconstruction = projection_error(decoder.lbs, displacements, rest_pts=rest,
                                                sample_masses=self.node_masses, ridge=1e-10)
        floor = per_vertex_l2_error(reconstruction, window.target_positions)

        return {
            'rollout_mean': float(distances.mean()),
            'rollout_max': float(distances.max()),
            'projection_mean': float(floor.mean()),
            'projection_max': float(floor.max()),
            'target_max': float(displacements.norm(dim=-1).max()),
        }

    def pretrain_projection(self, num_steps, horizon=None, log_every=25, verbose=True):
        r"""Fit the basis to full-order snapshots before any rollout, as an initializer.

        Minimizes the mass-weighted best-projection error over the training frames. Much cheaper
        per step than a rollout, and it gives the trajectory loss a basis that can at least
        *represent* the motion before being asked to reproduce its dynamics.

        Args:
            num_steps (int): Optimizer steps.
            horizon (int, optional): Frames to fit. Default: the whole training range.
            log_every (int, optional): Logging interval. Default: 25.
            verbose (bool, optional): Print progress. Default: True.

        Returns:
            list of dict: Per-log-step records.
        """
        window = self.dataset.window(horizon or self.dataset.split_frame(), 0)
        displacements = window.target_positions - self.dataset.rest_positions
        records = []

        for step in range(num_steps):
            self.optimizer.zero_grad(set_to_none=True)
            weights = self.skinning_mod.compute_skinning_weights(self.dataset.rest_positions)
            lbs = dense_lbs_matrix(self.dataset.rest_positions, weights)
            error, _, _ = projection_error(lbs, displacements, sample_masses=self.node_masses,
                                           ridge=1e-10)
            error.backward()
            if self.config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.skinning_mod.parameters(),
                                               self.config.grad_clip)
            self.optimizer.step()

            if step % log_every == 0 or step == num_steps - 1:
                record = {'step': step, 'projection_error': float(error)}
                records.append(record)
                if verbose:
                    print(f"  pretrain {step:5d}  projection MSE {record['projection_error']:.6e}")
        return records

    def train_data_free(self, num_steps, num_samples=1000, batch_size=10, learning_rate=1e-3,
                        elastic_coeff=1e-1, ortho_coeff=1e6, interp_schedule=True, grad_clip=0.0,
                        log_every=None, eval_horizon=4, verbose=True):
        r"""Train with the *original* data-free Simplicits objective only, as the baseline.

        This is the comparison the whole exercise is for: the same network, the same handle count,
        the same evaluation metric, but trained on randomized elastic energy plus weight
        orthogonality with no knowledge of the simulator. Evaluated with :meth:`evaluate`, so the
        held-out rollout numbers are directly comparable to :meth:`train`.

        The defaults deliberately mirror ``SimplicitsObject.create_with_mlp``: learning rate
        :math:`10^{-3}`, ``le_coeff`` :math:`10^{-1}`, ``lo_coeff`` :math:`10^{6}`, 1000 sample
        points, batch 10, no gradient clipping, and the linear-to-Neo-Hookean interpolation ramp
        ``en_interp = step / num_steps``. That ramp matters: without it the objective starts fully
        Neo-Hookean, and a short run then produces a *worse* field than initialization. Comparing
        against a mis-scheduled baseline would be worse than not comparing at all, so this method
        keeps its own optimizer rather than sharing the trainer's.

        Args:
            num_steps (int): Optimizer steps. The published recipe uses 10000; anything under a few
                thousand has not finished the interpolation ramp in any meaningful sense.
            num_samples (int, optional): Points per step. Default: 1000.
            batch_size (int, optional): Random handle configurations per step. Default: 10.
            learning_rate (float, optional): Adam learning rate. Default: 1e-3.
            elastic_coeff (float, optional): Elastic term weight. Default: 1e-1.
            ortho_coeff (float, optional): Orthogonality term weight. Default: 1e6.
            interp_schedule (bool, optional): Ramp ``en_interp`` from 0 to 1 over the run, as
                upstream does. Default: True.
            grad_clip (float, optional): Gradient-norm clip; 0 disables, as upstream. Default: 0.0.
            log_every (int, optional): Logging interval. Default: ``num_steps // 10``.
            eval_horizon (int, optional): Horizon for the held-out evaluation. Default: 4.
            verbose (bool, optional): Print progress. Default: True.

        Returns:
            list of dict: Per-log-step records, with the same ``eval_*`` keys :meth:`train` emits.
        """
        normalized = self.skinning_mod._offset_scale(self.points)
        yms, prs, rhos = self.dataset.material_at(self.points)
        num_samples = min(num_samples, self.points.shape[0])
        log_every = log_every or max(num_steps // 10, 1)
        optimizer = torch.optim.Adam(self.skinning_mod.parameters(), learning_rate)
        records = []

        for step in range(num_steps):
            optimizer.zero_grad(set_to_none=True)
            interp = float(step / num_steps) if interp_schedule else 1.0
            elastic, ortho = data_free_regularizer(self.skinning_mod, normalized, yms, prs, rhos,
                                                   self.dataset.total_volume,
                                                   num_samples=num_samples, batch_size=batch_size,
                                                   interp_step=interp, elastic_coeff=elastic_coeff,
                                                   ortho_coeff=ortho_coeff)
            loss = elastic + ortho
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.skinning_mod.parameters(), grad_clip)
            optimizer.step()

            if step % log_every == 0 or step == num_steps - 1:
                record = {'step': step, 'loss': float(loss), 'elastic': float(elastic),
                          'ortho': float(ortho), 'interp': interp}
                metrics = self.evaluate(self.dataset.eval_window(eval_horizon))
                record.update({f'eval_{k}': v for k, v in metrics.items()})
                records.append(record)
                if verbose:
                    print(f"  data-free {step:5d}  loss {record['loss']:.6e}  "
                          f"(elastic {record['elastic']:.3e} ortho {record['ortho']:.3e} "
                          f"interp {interp:.2f})  held-out rollout mean "
                          f"{metrics['rollout_mean']:.4e} m")
        return records

    def train(self, num_steps, horizon=4, horizon_schedule=None, stride=1, log_every=10,
              eval_every=50, verbose=True):
        r"""The training loop, cycling over training-range windows.

        Args:
            num_steps (int): Optimizer steps.
            horizon (int, optional): Window length. Default: 4.
            horizon_schedule (list of tuple, optional): ``[(step, horizon), ...]`` curriculum,
                applied when the step count reaches each entry. Short horizons first: a long
                unrolled window from an untrained basis produces a large, badly conditioned
                gradient. Default: None.
            stride (int, optional): Stride between window starts. Default: 1.
            log_every (int, optional): Logging interval. Default: 10.
            eval_every (int, optional): Held-out evaluation interval. Default: 50.
            verbose (bool, optional): Print progress. Default: True.

        Returns:
            list of dict: The training history.
        """
        schedule = sorted(horizon_schedule or [], key=lambda item: item[0])
        windows = self.dataset.train_windows(horizon, stride=stride)
        start = time.time()

        for step in range(num_steps):
            for threshold, new_horizon in schedule:
                if step == threshold and new_horizon != horizon:
                    horizon = new_horizon
                    windows = self.dataset.train_windows(horizon, stride=stride)
                    if verbose:
                        print(f'  step {step}: horizon -> {horizon} ({len(windows)} windows)')

            record = self.train_step(windows[step % len(windows)])
            record['step'] = step
            record['elapsed'] = time.time() - start

            if verbose and (step % log_every == 0 or step == num_steps - 1):
                print(f"  step {step:5d}  loss {record['loss']:.6e}  "
                      f"pos {record['position']:.6e}  |g| {record['grad_norm']:.3e}  "
                      f"h={record['horizon']} t0={record['start_frame']}")

            if eval_every > 0 and (step % eval_every == 0 or step == num_steps - 1):
                metrics = self.evaluate(self.dataset.eval_window(horizon))
                record.update({f'eval_{k}': v for k, v in metrics.items()})
                if verbose:
                    print(f"    held-out rollout mean {metrics['rollout_mean']:.4e} m  "
                          f"max {metrics['rollout_max']:.4e} m  |  projection floor mean "
                          f"{metrics['projection_mean']:.4e} m  |  motion scale "
                          f"{metrics['target_max']:.4e} m")

            self.history.append(record)
        return self.history


def _default_beam_path(resolution, frames):
    tag = 'x'.join(str(r) for r in resolution)
    return os.path.join('data', f'fom_beam_{tag}_{frames}frames.pth')


def _ensure_beam_data(resolution, frames, device, dtype, verbose=True):
    """Load the beam trajectory, generating it first if it is not on disk."""
    from .data_gen.gen_fom_beam import generate_trajectory, load_trajectory

    path = _default_beam_path(resolution, frames)
    if os.path.exists(path):
        if verbose:
            print(f'loading {path}')
        return load_trajectory(path)

    if verbose:
        print(f'{path} not found; generating')
    trajectory = generate_trajectory(num_frames=frames, resolution=resolution, device=device,
                                     dtype=dtype, verbose=verbose)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(trajectory, path)
    return trajectory


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--beam', action='store_true', help='use the cantilever-beam scenario')
    parser.add_argument('--resolution', type=int, nargs=3, default=[10, 3, 3],
                        help='full-order grid resolution')
    parser.add_argument('--frames', type=int, default=40, help='frames in the full-order data')
    parser.add_argument('--handles', type=int, default=8, help='number of handles, including the '
                                                               'constant one')
    parser.add_argument('--layer-width', type=int, default=64)
    parser.add_argument('--layers', type=int, default=3)
    parser.add_argument('--num-qp', type=int, default=512)
    parser.add_argument('--horizon', type=int, default=4)
    parser.add_argument('--steps', type=int, default=200,
                        help='simulation-in-the-loop training steps')
    parser.add_argument('--data-free-steps', type=int, default=4000,
                        help='data-free baseline steps; upstream uses 10000')
    parser.add_argument('--newton-steps', type=int, default=4)
    parser.add_argument('--pretrain', type=int, default=0,
                        help='snapshot-projection steps before trajectory training')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--data-free-coeff', type=float, default=0.0)
    parser.add_argument('--bptt-window', type=int, default=None)
    parser.add_argument('--mode', type=str, default='sim_in_loop',
                        choices=['sim_in_loop', 'data_free', 'both'],
                        help="'both' trains two copies of the same network from the same seed and "
                             'reports their held-out rollout error side by side')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--float32', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    if not args.beam:
        parser.error('only --beam is wired up so far')

    dtype = torch.float32 if args.float32 else torch.float64
    torch.manual_seed(args.seed)

    trajectory = _ensure_beam_data(tuple(args.resolution), args.frames, args.device, dtype)
    dataset = TrajectoryDataset([trajectory], device=args.device, dtype=dtype)

    rest = dataset.rest_positions

    def make_trainer():
        # Reseeded per trainer so the two modes start from the same theta and the same quadrature.
        torch.manual_seed(args.seed)
        skinning_mod = SimplicitsMLP(3, args.layer_width, args.handles, args.layers,
                                     bb_min=rest.min(dim=0).values.cpu(),
                                     bb_max=rest.max(dim=0).values.cpu())
        skinning_mod = skinning_mod.to(device=args.device, dtype=dtype)
        config = TrainerConfig(num_qp=args.num_qp, num_newton_steps=args.newton_steps,
                               learning_rate=args.lr, data_free_coeff=args.data_free_coeff,
                               bptt_window=args.bptt_window)
        generator = torch.Generator(device=args.device).manual_seed(args.seed)
        return SimInLoopTrainer(skinning_mod, dataset, config, generator=generator)

    reference = make_trainer()
    print(f'{dataset.num_frames()} frames, {rest.shape[0]} nodes, '
          f'{args.handles} handles ({12 * args.handles} reduced dofs), '
          f'{args.num_qp} quadrature points ({reference.num_pinned} pinned)')
    initial = reference.evaluate(dataset.eval_window(args.horizon))
    print(f"at initialization: held-out rollout mean {initial['rollout_mean']:.4e} m, "
          f"projection floor {initial['projection_mean']:.4e} m, "
          f"motion scale {initial['target_max']:.4e} m")

    results = {}
    if args.mode in ('sim_in_loop', 'both'):
        trainer = reference if args.mode == 'sim_in_loop' else make_trainer()
        if args.pretrain > 0:
            trainer.pretrain_projection(args.pretrain)
        trainer.train(args.steps, horizon=args.horizon)
        results['sim_in_loop'] = trainer.evaluate(dataset.eval_window(args.horizon))

    if args.mode in ('data_free', 'both'):
        trainer = make_trainer()
        trainer.train_data_free(args.data_free_steps, eval_horizon=args.horizon)
        results['data_free'] = trainer.evaluate(dataset.eval_window(args.horizon))

    print(f'\n{"mode":<14} {"rollout mean":>14} {"rollout max":>14} {"projection floor":>18}')
    for name, metrics in results.items():
        print(f"{name:<14} {metrics['rollout_mean']:14.4e} {metrics['rollout_max']:14.4e} "
              f"{metrics['projection_mean']:18.4e}")
    print(f"{'(motion scale)':<14} {initial['target_max']:14.4e}")


if __name__ == '__main__':
    main()
