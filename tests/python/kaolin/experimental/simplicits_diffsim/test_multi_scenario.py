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

"""Training on a pool of load scenarios rather than a single trajectory.

The failure these guard against is silent. Before per-scenario forces existed, ``TrajectoryDataset``
collapsed ``controls`` to trajectory 0's and the trainer baked that one gravity vector into a single
``forces.Gravity`` at construction, so a dataset of differently-loaded scenarios would train every
one of them against trajectory 0's load and report a perfectly plausible decreasing loss. Nothing
would raise. Hence
:func:`test_each_scenario_falls_along_its_own_gravity`, which checks the physics rather than the
plumbing, and :func:`test_shared_controls_must_match`, which turns the unsupported half of the same
problem into an exception.
"""

import pytest
import torch

from kaolin.physics.simplicits.network import SimplicitsMLP
from kaolin.experimental.simplicits_diffsim.data_gen.gen_fom_beam import (generate_trajectory,
                                                                         gravity_sweep_scenarios)
from kaolin.experimental.simplicits_diffsim.dataset import TrajectoryDataset
from kaolin.experimental.simplicits_diffsim.trainer import SimInLoopTrainer, TrainerConfig

RESOLUTION = (4, 2, 2)
NUM_FRAMES = 6
NUM_HANDLES = 4

# Mutually orthogonal / opposed loads, so "did this scenario's gravity reach the simulator?" has an
# unambiguous answer in the sign of the resulting displacement.
GRAVITIES = ((0.0, 9.8, 0.0), (0.0, 0.0, 9.8), (0.0, -9.8, 0.0))


@pytest.fixture(scope='module')
def scenario_trajectories():
    """One tiny full-order trajectory per gravity direction."""
    return [generate_trajectory(num_frames=NUM_FRAMES, resolution=RESOLUTION, device='cuda',
                                dtype=torch.float64, verbose=False, scenario={'gravity': gravity})
            for gravity in GRAVITIES]


@pytest.fixture(scope='module')
def multi_dataset(scenario_trajectories):
    return TrajectoryDataset(scenario_trajectories, device='cuda', dtype=torch.float64,
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
def test_shared_controls_must_match(device, scenario_trajectories):
    """Controls the trainer resolves only once must be rejected when they disagree.

    Gravity is per-scenario and fine. Material, time step and the pin predicate are folded into the
    quadrature weights, the Lame parameters and :math:`B^T M B` at construction, so a dataset that
    varied them would be quietly simulating the wrong thing.
    """
    dataset = TrajectoryDataset(scenario_trajectories, device=device, dtype=torch.float64)
    assert dataset.num_trajectories == len(GRAVITIES)

    for key, value in (('youngs_modulus', 2e5), ('density', 250.0), ('timestep', 0.01),
                       ('pin_threshold', 0.5)):
        first, second = scenario_trajectories[0], scenario_trajectories[1]
        tampered = {**second, 'controls': {**second['controls'], key: value}}
        with pytest.raises(ValueError, match=key):
            TrajectoryDataset([first, tampered], device=device, dtype=torch.float64)


@pytest.mark.parametrize('device', ['cuda'])
def test_each_scenario_falls_along_its_own_gravity(device, multi_dataset):
    """Every scenario's rollout must move along its *own* :math:`-g`, not trajectory 0's.

    ``SimplicitsScene``'s sign convention is an energy :math:`+ m g \\cdot x`, so the force is along
    :math:`-g`. All three windows start from rest, so the load is the only thing that differs
    between these rollouts: if the per-scenario forces were not wired through, all three would move
    the same way and the orthogonal and opposed scenarios would fail.
    """
    trainer = _trainer(multi_dataset)
    model, decoder = trainer.build()
    rest = multi_dataset.rest_positions

    displacements = []
    for index in range(multi_dataset.num_trajectories):
        window = multi_dataset.window(horizon=2, start_frame=0, trajectory_index=index)
        assert window.starts_at_rest
        with torch.no_grad():
            predicted = trainer.predict(window, model, decoder)
        displacement = (predicted[-1] - rest).mean(dim=0)
        gravity = multi_dataset.gravity_at(index)
        along = float(displacement @ (-gravity / gravity.norm()))
        assert along > 0.0, f'scenario {index} moved {along:.3e} m along -g'
        displacements.append(displacement)

    # The opposed pair must genuinely oppose, which no shared-gravity bug could produce.
    assert float(displacements[0] @ displacements[2]) < 0.0


@pytest.mark.parametrize('device', ['cuda'])
def test_with_pt_forces_shares_assembly_and_keeps_gradient(device, multi_dataset):
    """Rebinding forces must reuse B, dF/dz and B^T M B, and still backprop to theta.

    Sharing the assembly is what makes many scenarios per step affordable; keeping the autograd
    history is what makes the shared assembly usable for training rather than only for evaluation.
    """
    trainer = _trainer(multi_dataset)
    model, _ = trainer.build()
    rebound = model.with_pt_forces(trainer.scenario_forces[1])

    assert rebound.lbs is model.lbs
    assert rebound.dFdz is model.dFdz
    assert rebound.reduced_mass is model.reduced_mass
    assert rebound.pt_forces is not model.pt_forces
    assert rebound.pt_forces[0] is trainer.scenario_forces[1][0]

    torch.manual_seed(0)
    coords = 1e-3 * torch.randn(model.num_reduced_dofs, device=device, dtype=torch.float64)
    parameters = list(trainer.skinning_mod.parameters())
    grads = torch.autograd.grad(rebound.potential_energy(coords), parameters, allow_unused=True)
    assert any(g is not None and float(g.abs().sum()) > 0.0 for g in grads)


@pytest.mark.parametrize('device', ['cuda'])
def test_predicting_a_foreign_scenario_raises(device, multi_dataset, scenario_trajectories):
    """A trainer must refuse a window from a scenario set it was not built for.

    Loads, quadrature and masses are resolved once from the construction dataset, so evaluating a
    field on a *larger* set means rebuilding the trainer on that set and loading the state dict --
    not reusing this one, which would silently apply the wrong load.
    """
    subset = TrajectoryDataset(scenario_trajectories[:1], device=device, dtype=torch.float64,
                               train_fraction=0.5)
    trainer = _trainer(subset)
    foreign = multi_dataset.window(horizon=2, start_frame=0, trajectory_index=2)
    with pytest.raises(IndexError, match='scenario 2'):
        trainer.predict(foreign)


@pytest.mark.parametrize('device', ['cuda'])
def test_pooled_windows_cover_every_scenario(device, multi_dataset):
    """The training pool must span all scenarios, and each window must know which it came from."""
    windows = multi_dataset.all_train_windows(horizon=2)
    per_scenario = multi_dataset.train_windows(horizon=2, trajectory_index=0)

    assert len(windows) == len(per_scenario) * multi_dataset.num_trajectories
    assert {w.trajectory_index for w in windows} == set(range(multi_dataset.num_trajectories))
    assert all(w.horizon == 2 for w in windows)

    snapshots = multi_dataset.train_snapshots()
    expected = sum(multi_dataset.split_frame(i) for i in range(multi_dataset.num_trajectories))
    assert snapshots.shape == (expected, multi_dataset.rest_positions.shape[0], 3)
    assert len(multi_dataset.all_eval_windows(2)) == multi_dataset.num_trajectories


@pytest.mark.parametrize('device', ['cuda'])
def test_evaluate_all_aggregates_every_scenario(device, multi_dataset):
    """Aggregate evaluation must report the worst scenario, not only the mean over them."""
    trainer = _trainer(multi_dataset)
    metrics = trainer.evaluate_all(horizon=2)

    assert metrics['num_scenarios'] == multi_dataset.num_trajectories
    assert len(metrics['per_scenario']) == multi_dataset.num_trajectories
    assert 0 <= metrics['worst_scenario'] < multi_dataset.num_trajectories
    assert metrics['worst_rollout_mean'] >= metrics['rollout_mean'] - 1e-12
    for key in ('rollout_mean', 'rollout_max', 'projection_mean', 'target_max'):
        assert torch.isfinite(torch.tensor(metrics[key])) and metrics[key] >= 0.0
    assert metrics['target_max'] > 0.0


@pytest.mark.parametrize('device', ['cuda'])
def test_training_visits_multiple_scenarios_and_decreases_pooled_loss(device, multi_dataset):
    """Training must draw from more than one scenario and reduce the loss over the pool."""
    trainer = _trainer(multi_dataset)
    history = trainer.train(12, horizon=2, log_every=10 ** 9, eval_every=0, verbose=False)

    assert len({record['trajectory_index'] for record in history}) > 1
    assert all(torch.isfinite(torch.tensor(record['loss'])) for record in history)

    first_epoch = sum(r['loss'] for r in history[:6]) / 6
    last_epoch = sum(r['loss'] for r in history[-6:]) / 6
    assert last_epoch < first_epoch, f'pooled loss went {first_epoch:.3e} -> {last_epoch:.3e}'


@pytest.mark.parametrize('device', ['cuda'])
def test_pretraining_fits_snapshots_from_all_scenarios(device, multi_dataset):
    """Snapshot pretraining must consume every scenario's training frames."""
    trainer = _trainer(multi_dataset)
    records = trainer.pretrain_projection(30, log_every=10, verbose=False)
    assert records[-1]['projection_error'] < records[0]['projection_error']


def test_gravity_sweep_scenarios_are_distinct_and_correctly_scaled():
    """The sweep must produce unit directions at the requested magnitudes, without duplicates."""
    magnitudes = (4.9, 9.8)
    scenarios = gravity_sweep_scenarios(num_directions=6, magnitudes=magnitudes)
    assert len(scenarios) == 6 * len(magnitudes)

    vectors = torch.tensor([s['gravity'] for s in scenarios], dtype=torch.float64)
    norms = vectors.norm(dim=-1)
    for magnitude in magnitudes:
        assert int((norms - magnitude).abs().lt(1e-9).sum()) == 6

    # Well-spread means no two directions coincide.
    directions = vectors[:6] / norms[:6, None]
    cosines = directions @ directions.T
    assert float(cosines.triu(diagonal=1).max()) < 0.95
