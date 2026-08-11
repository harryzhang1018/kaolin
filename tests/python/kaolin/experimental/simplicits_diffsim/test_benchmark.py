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

"""Smoke tests for the timing helpers -- that they measure, and report, the right thing."""

import pytest
import torch

from kaolin.experimental.simplicits_diffsim.benchmark import (StepTiming, time_full_order,
                                                             time_reduced)
from kaolin.experimental.simplicits_diffsim.data_gen.fem_hex import (FullOrderNeohookeanSolver,
                                                                    HexGrid)


def test_step_timing_derives_fps_and_rtf():
    """Real-time factor is dt/step-time; fps is its reciprocal per frame."""
    timing = StepTiming('x', seconds_per_frame=0.01, timestep=0.05, num_dofs=96)
    assert timing.fps == pytest.approx(100.0)
    assert timing.rtf == pytest.approx(5.0)
    assert '96' in str(timing)


@pytest.mark.parametrize('device', ['cuda'])
def test_timings_are_positive_and_labelled(device, make_matched_pair):
    """Both paths must return a positive, finite measurement carrying its own dof count."""
    grid = HexGrid((0.0, 0.0, 0.0), (1.0, 0.25, 0.25), (3, 1, 1), device=device,
                   dtype=torch.float64)
    pinned = torch.nonzero(grid.nodes[:, 0] >= 0.99, as_tuple=False).squeeze(1)
    solver = FullOrderNeohookeanSolver(grid, 1e5, 0.45, 500.0, timestep=0.05,
                                       pinned_nodes=pinned)
    full = time_full_order(solver, num_frames=2, warmup=1)
    assert full.seconds_per_frame > 0 and full.num_dofs == int(solver.free_dofs.numel())

    pair = make_matched_pair(device, num_qp=64, num_handles=3)
    reduced = time_reduced(pair.model, pair.timestep, num_frames=2, warmup=1)
    assert reduced.seconds_per_frame > 0
    assert reduced.num_dofs == pair.model.num_reduced_dofs
    assert reduced.rtf == pytest.approx(pair.timestep / reduced.seconds_per_frame)


@pytest.mark.parametrize('device', ['cuda'])
def test_timed_step_does_not_disturb_the_model(device, make_matched_pair):
    """Timing runs under no_grad and must leave the model's operators untouched."""
    pair = make_matched_pair(device, num_qp=64, num_handles=3)
    before = pair.model.lbs.clone()
    time_reduced(pair.model, pair.timestep, num_frames=2, warmup=1)
    assert torch.equal(pair.model.lbs, before)
