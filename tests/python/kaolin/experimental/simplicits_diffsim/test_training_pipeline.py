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

"""End-to-end wiring: generate full-order data, slice it, train through the reduced simulator.

Deliberately tiny and fast. The scientific question -- whether trajectory training beats the
data-free objective -- is not something a unit test can answer; what these check is that the
pipeline is connected, that the losses actually decrease, and that the projection floor is reported
alongside rollout error so the numbers stay interpretable.
"""

import pytest
import torch

from kaolin.physics.simplicits.network import SimplicitsMLP
from kaolin.experimental.simplicits_diffsim.data_gen.gen_fom_beam import generate_trajectory
from kaolin.experimental.simplicits_diffsim.dataset import TrajectoryDataset
from kaolin.experimental.simplicits_diffsim.projection import pod_basis, projection_error
from kaolin.experimental.simplicits_diffsim.trainer import SimInLoopTrainer, TrainerConfig

RESOLUTION = (4, 2, 2)
NUM_FRAMES = 6
NUM_HANDLES = 4


@pytest.fixture(scope='module')
def tiny_dataset():
    """A 4x2x2 beam over 6 frames -- enough to exercise every code path in about a second."""
    trajectory = generate_trajectory(num_frames=NUM_FRAMES, resolution=RESOLUTION, device='cuda',
                                    dtype=torch.float64, verbose=False)
    return TrajectoryDataset([trajectory], device='cuda', dtype=torch.float64,
                             train_fraction=0.5)


def _trainer(dataset, **overrides):
    torch.manual_seed(0)
    rest = dataset.rest_positions
    skinning_mod = SimplicitsMLP(3, 16, NUM_HANDLES, 1, bb_min=rest.min(dim=0).values.cpu(),
                                 bb_max=rest.max(dim=0).values.cpu())
    skinning_mod = skinning_mod.to(device=rest.device, dtype=rest.dtype)
    config = TrainerConfig(num_qp=128, num_newton_steps=3, learning_rate=1e-3, **overrides)
    return SimInLoopTrainer(skinning_mod, dataset, config)


@pytest.mark.parametrize('device', ['cuda'])
def test_dataset_windows_and_split(device, tiny_dataset):
    """Windows must stay inside the trajectory and the split must hold frames back."""
    dataset = tiny_dataset
    assert dataset.num_frames() == NUM_FRAMES
    split = dataset.split_frame()
    assert 0 < split < NUM_FRAMES

    windows = dataset.train_windows(horizon=2)
    assert windows and all(w.horizon == 2 for w in windows)
    assert all(w.start_frame + w.horizon <= split for w in windows)
    assert windows[0].starts_at_rest

    held_out = dataset.eval_window()
    assert held_out.start_frame == split
    assert not held_out.starts_at_rest
    assert held_out.start_frame + held_out.horizon == NUM_FRAMES

    with pytest.raises(ValueError):
        dataset.window(horizon=NUM_FRAMES + 1, start_frame=0)


@pytest.mark.parametrize('device', ['cuda'])
def test_projected_initial_state_beats_rest_for_midtrajectory_windows(device, tiny_dataset):
    """A window starting mid-trajectory must be initialized by projection, not from rest."""
    trainer = _trainer(tiny_dataset)
    _, decoder = trainer.build()
    window = tiny_dataset.eval_window()

    coords, velocity = window.initial_reduced_state(decoder.lbs, trainer.node_masses)
    assert coords.abs().max() > 0 and velocity.abs().max() > 0

    projected = tiny_dataset.rest_positions + (decoder.lbs @ coords).reshape(-1, 3)
    target = tiny_dataset.rest_positions + window.initial_displacements
    from_rest = (tiny_dataset.rest_positions - target).norm(dim=-1).mean()
    from_projection = (projected - target).norm(dim=-1).mean()
    assert from_projection < from_rest


@pytest.mark.parametrize('device', ['cuda'])
def test_train_step_decreases_loss_and_moves_parameters(device, tiny_dataset):
    """A handful of steps must reduce the window loss and actually update theta."""
    trainer = _trainer(tiny_dataset)
    before = torch.nn.utils.parameters_to_vector(trainer.skinning_mod.parameters()).detach().clone()
    window = tiny_dataset.train_windows(horizon=2)[0]

    first = trainer.train_step(window)
    assert torch.isfinite(torch.tensor(first['loss'])) and first['grad_norm'] > 0
    losses = [first['loss']]
    for _ in range(9):
        losses.append(trainer.train_step(window)['loss'])

    after = torch.nn.utils.parameters_to_vector(trainer.skinning_mod.parameters()).detach()
    assert (after - before).norm() > 0
    assert losses[-1] < losses[0], f'loss went {losses[0]:.3e} -> {losses[-1]:.3e}'


@pytest.mark.parametrize('device', ['cuda'])
def test_evaluate_reports_rollout_and_projection_floor(device, tiny_dataset):
    """Evaluation must return finite rollout error, a projection floor, and the motion scale."""
    trainer = _trainer(tiny_dataset)
    metrics = trainer.evaluate(tiny_dataset.eval_window())
    for key in ('rollout_mean', 'rollout_max', 'projection_mean', 'projection_max', 'target_max'):
        assert key in metrics and metrics[key] >= 0.0
        assert torch.isfinite(torch.tensor(metrics[key]))
    assert metrics['target_max'] > 0.0
    assert metrics['projection_mean'] <= metrics['projection_max']


@pytest.mark.parametrize('device', ['cuda'])
def test_projection_pretraining_reduces_projection_error(device, tiny_dataset):
    """Snapshot pretraining must reduce the quantity it optimizes."""
    trainer = _trainer(tiny_dataset)
    records = trainer.pretrain_projection(30, log_every=10, verbose=False)
    assert records[-1]['projection_error'] < records[0]['projection_error']


@pytest.mark.parametrize('device', ['cuda'])
def test_pod_upper_bounds_the_learned_basis(device, tiny_dataset):
    """POD of the same rank must project the snapshots at least as well as the skinning basis.

    POD is optimal over *all* linear subspaces of that dimension, so this inequality is a sanity
    check on the projection code rather than a statement about training: if a learned basis ever
    beat POD at equal rank, the projection or the basis size would be wrong.
    """
    trainer = _trainer(tiny_dataset)
    _, decoder = trainer.build()
    displacements = tiny_dataset.full_window().target_positions - tiny_dataset.rest_positions

    learned_error, _, _ = projection_error(decoder.lbs, displacements,
                                           sample_masses=trainer.node_masses, ridge=1e-12)
    num_modes = min(decoder.lbs.shape[1], displacements.shape[0])
    basis, spectrum = pod_basis(displacements, num_modes, sample_masses=trainer.node_masses)
    pod_error, _, _ = projection_error(basis, displacements, sample_masses=trainer.node_masses,
                                       ridge=1e-12)

    assert spectrum.numel() > 0 and float(spectrum[0]) > 0
    assert float(pod_error) <= float(learned_error) * (1.0 + 1e-9)


@pytest.mark.parametrize('device', ['cuda'])
def test_truncated_bptt_config_runs(device, tiny_dataset):
    """The truncated-BPTT path must produce a usable gradient through the trainer as well."""
    trainer = _trainer(tiny_dataset, bptt_window=1)
    record = trainer.train_step(tiny_dataset.train_windows(horizon=3)[0])
    assert record['grad_norm'] > 0 and torch.isfinite(torch.tensor(record['loss']))


@pytest.mark.parametrize('device', ['cuda'])
def test_data_free_baseline_trains(device, tiny_dataset):
    """The data-free baseline must run and reduce its own objective.

    It also pins down a shape contract: ``kaolin.physics.simplicits.losses.loss_elastic``
    broadcasts the Lame parameters as ``mus.expand(num_samples, batch_size)``, so it needs a
    trailing singleton dimension that this package's ``(num_pts,)`` convention does not have.
    """
    trainer = _trainer(tiny_dataset)
    records = trainer.train_data_free(20, num_samples=64, batch_size=4, log_every=10,
                                      eval_horizon=2, verbose=False)
    assert len(records) >= 2
    assert records[-1]['loss'] < records[0]['loss']
    assert all(torch.isfinite(torch.tensor(r['eval_rollout_mean'])) for r in records)


@pytest.mark.parametrize('device', ['cuda'])
def test_ortho_term_matches_upstream_in_float32(device):
    """The locally rebuilt orthogonality term must equal upstream's, and work in float64."""
    from kaolin.physics.simplicits.losses import loss_ortho
    from kaolin.experimental.simplicits_diffsim.losses import ortho_term

    torch.manual_seed(0)
    weights = torch.rand(64, 5, device=device, dtype=torch.float32)
    assert abs(float(ortho_term(weights)) - float(loss_ortho(weights))) < 1e-6

    # The reason it exists: upstream's float32 identity breaks the float64 backward pass.
    double = weights.double().requires_grad_(True)
    torch.autograd.grad(ortho_term(double), double)
    with pytest.raises(RuntimeError):
        torch.autograd.grad(loss_ortho(double), double)
