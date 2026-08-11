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

"""Smoke tests for the trajectory renderer."""

import os

import pytest
import torch

from kaolin.experimental.simplicits_diffsim.data_gen.fem_hex import HexGrid
from kaolin.experimental.simplicits_diffsim.viz import _permutation_for, render_trajectory


def test_up_axis_permutation_puts_gravity_axis_vertical():
    """The permutation must move the requested world axis to the plot's vertical slot."""
    assert _permutation_for(1) == [0, 2, 1]
    assert _permutation_for(2) == [0, 1, 2]
    assert _permutation_for(0) == [1, 2, 0]
    for axis in range(3):
        assert sorted(_permutation_for(axis)) == [0, 1, 2]
        assert _permutation_for(axis)[2] == axis


@pytest.mark.parametrize('with_reference', [False, True])
def test_render_writes_a_playable_file(tmp_path, with_reference):
    """Rendering must produce a non-trivial file, with and without the ghost overlay."""
    grid = HexGrid((0.0, 0.0, 0.0), (1.0, 0.25, 0.25), (3, 1, 1), device='cpu',
                   dtype=torch.float64)
    positions = torch.stack([grid.nodes + 0.01 * frame for frame in range(4)], dim=0)
    reference = positions + 0.005 if with_reference else None

    out = render_trajectory(str(tmp_path / 'clip.mp4'), positions, grid.surface_faces(),
                            reference=reference, title='test', fps=4, dpi=60, dt=0.05)
    assert os.path.exists(out)
    assert os.path.getsize(out) > 2000
