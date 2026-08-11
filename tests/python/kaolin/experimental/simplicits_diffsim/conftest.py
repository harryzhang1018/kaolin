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

"""Shared setup for the differentiable-twin equivalence tests.

Builds a small ``SimplicitsScene`` (the Warp reference) together with a ``ReducedModel`` (the
torch twin) that is fed the scene's *own* baked weights. That isolates what these tests are
about -- the assembly and the step -- from any difference in how the weights were evaluated.

The scene is deliberately configured with ``normalize_weights_by_samples=False`` and
``apply_qr=False`` so that both sides use the same reduced coordinates, and with
``newton_hessian_regularizer=0``/``direct_solve=True`` so the linear solve is exact.
"""

import pytest
import torch
import warp as wp

from kaolin.physics.materials.material_utils import to_lame
from kaolin.physics.simplicits import SimplicitsObject, SimplicitsScene
from kaolin.physics.simplicits.network import SimplicitsMLP
from kaolin.experimental.simplicits_diffsim import forces
from kaolin.experimental.simplicits_diffsim.reduced_model import ReducedModel

NUM_QP = 256
NUM_HANDLES = 4
TIMESTEP = 0.05
FLOOR_HEIGHT = -1.0
FLOOR_AXIS = 1
FLOOR_PENALTY = 10000.0
BDRY_PENALTY = 10000.0
GRAVITY = (0.0, 9.8, 0.0)
PIN_THRESHOLD = 0.8


class MatchedPair:
    """A Warp scene and the torch twin built from its baked weights."""

    def __init__(self, scene, model, skinning_mod, sim_object, pinned_positions, pinned_indices):
        self.scene = scene
        self.model = model
        self.skinning_mod = skinning_mod
        self.sim_object = sim_object
        self.pinned_positions = pinned_positions
        self.pinned_indices = pinned_indices
        self.timestep = TIMESTEP

    def warp_residual(self, reduced_coords, reduced_coords_prev, reduced_velocity):
        """Reference residual from ``SimplicitsScene._newton_G``, as a torch tensor."""
        out = self.scene._newton_G(wp.from_torch(reduced_coords.contiguous()),
                                   wp.from_torch(reduced_coords_prev.contiguous()),
                                   wp.from_torch(reduced_velocity.contiguous()),
                                   self.scene.sim_B, self.scene.sim_BMB, self.timestep)
        return wp.to_torch(out).clone().flatten()

    def warp_hessian(self, reduced_coords):
        """Reference Hessian from ``SimplicitsScene._newton_H``, densified."""
        from kaolin.physics.utils import _wp_bsr_to_torch_bsr
        bsr = self.scene._newton_H(wp.from_torch(reduced_coords.contiguous()),
                                   self.scene.sim_B, self.scene.sim_BMB, self.timestep)
        return _wp_bsr_to_torch_bsr(bsr).to_dense().clone()

    def warp_energy(self, reduced_coords, reduced_coords_prev, reduced_velocity):
        """Reference incremental potential from ``SimplicitsScene._newton_E``."""
        return float(self.scene._newton_E(wp.from_torch(reduced_coords.contiguous()),
                                          wp.from_torch(reduced_coords_prev.contiguous()),
                                          wp.from_torch(reduced_velocity.contiguous()),
                                          self.scene.sim_B, self.timestep))


def build_matched_pair(device, num_qp=NUM_QP, num_handles=NUM_HANDLES, layer_width=8, num_layers=1,
                       seed=0, with_floor=True, with_pins=True, max_newton_steps=10):
    """Build a Warp scene and a torch twin sharing weights, points and material."""
    torch.manual_seed(seed)
    dtype = torch.float32

    skinning_mod = SimplicitsMLP(3, layer_width, num_handles, num_layers,
                                 bb_min=torch.zeros(3), bb_max=torch.ones(3)).to(device)
    pts = torch.rand(4 * num_qp, 3, device=device, dtype=dtype)
    sim_obj = SimplicitsObject(pts=pts,
                               yms=torch.full((4 * num_qp,), 1e5, device=device, dtype=dtype),
                               prs=torch.full((4 * num_qp,), 0.45, device=device, dtype=dtype),
                               rhos=torch.full((4 * num_qp,), 500.0, device=device, dtype=dtype),
                               appx_vol=torch.tensor(1.0, device=device, dtype=dtype),
                               skinning_mod=skinning_mod)

    scene = SimplicitsScene(device=device, timestep=TIMESTEP, max_newton_steps=max_newton_steps)
    scene.newton_hessian_regularizer = 0.0
    scene.direct_solve = True
    scene.add_object(sim_obj, num_qp=num_qp, normalize_weights_by_samples=False, apply_qr=False)
    scene.set_scene_gravity(torch.tensor(GRAVITY))
    if with_floor:
        scene.set_scene_floor(floor_height=FLOOR_HEIGHT, floor_axis=FLOOR_AXIS,
                              floor_penalty=FLOOR_PENALTY, flip_floor=False)
    pinned_positions, pinned_indices = None, None
    if with_pins:
        pinned_positions = scene.set_object_boundary_condition(
            0, 'right', lambda x: x[:, 0] >= PIN_THRESHOLD, bdry_penalty=BDRY_PENALTY)

    sim_object = scene.get_object(0)
    if with_pins:
        pinned_indices = torch.nonzero(sim_object.pts[:, 0] >= PIN_THRESHOLD,
                                       as_tuple=False).squeeze(1)

    mus, lams = to_lame(sim_object.yms, sim_object.prs)
    pt_forces = [forces.Gravity(torch.tensor(GRAVITY, device=device, dtype=dtype),
                                sim_object.rhos, sim_object.sample_vols)]
    if with_floor:
        pt_forces.append(forces.Floor(FLOOR_HEIGHT, FLOOR_AXIS, False, FLOOR_PENALTY))
    if with_pins:
        pt_forces.append(forces.Boundary(pinned_indices, pinned_positions, BDRY_PENALTY))

    model = ReducedModel(sim_object.pts, sim_object.skinning_weights, sim_object.dwdx,
                         mus, lams, sim_object.sample_vols, sim_object.sample_masses,
                         pt_forces=pt_forces, reparameterize_lame=True)

    return MatchedPair(scene, model, skinning_mod, sim_object, pinned_positions, pinned_indices)


def build_float64_twin(pair):
    """Rebuild the torch twin of ``pair`` in double precision.

    ``SimplicitsScene`` hardcodes ``self.dtype = torch.float32``, so Warp comparisons are
    float32-limited. Anything compared against finite differences instead must run in float64,
    where a central difference actually has a usable window.
    """
    sim_object = pair.sim_object
    mus, lams = to_lame(sim_object.yms.double(), sim_object.prs.double())
    pt_forces = []
    for force in pair.model.pt_forces:
        if isinstance(force, forces.Gravity):
            pt_forces.append(forces.Gravity(force.gravity.double(), sim_object.rhos.double(),
                                            sim_object.sample_vols.double(), force.coeff))
        elif isinstance(force, forces.Floor):
            pt_forces.append(forces.Floor(force.floor_height, force.floor_axis, force.flip_floor,
                                          force.coeff))
        else:
            pt_forces.append(forces.Boundary(force.pinned_indices,
                                             force.pinned_positions.double(), force.coeff))
    return ReducedModel(sim_object.pts.double(), sim_object.skinning_weights.double(),
                        sim_object.dwdx.double(), mus, lams, sim_object.sample_vols.double(),
                        sim_object.sample_masses.double(), pt_forces=pt_forces,
                        reparameterize_lame=True)


@pytest.fixture(scope='module')
def make_matched_pair():
    """Factory fixture exposing :func:`build_matched_pair` without a cross-module import."""
    return build_matched_pair


@pytest.fixture(scope='module')
def make_float64_twin():
    """Factory fixture exposing :func:`build_float64_twin`."""
    return build_float64_twin


@pytest.fixture(scope='module')
def make_reduced_state():
    """Factory fixture exposing :func:`random_reduced_state`."""
    return random_reduced_state


@pytest.fixture(scope='module')
def matched_pair():
    """Warp scene plus torch twin, with gravity, floor and pins."""
    return build_matched_pair('cuda')


def random_reduced_state(pair, seed=1, scale=0.05):
    """A small random reduced state, previous state and velocity."""
    generator = torch.Generator(device=pair.model.lbs.device).manual_seed(seed)
    size = (pair.model.num_reduced_dofs,)
    kwargs = dict(device=pair.model.lbs.device, dtype=pair.model.lbs.dtype, generator=generator)
    coords = (torch.rand(size, **kwargs) - 0.5) * scale
    prev = torch.zeros(size, device=pair.model.lbs.device, dtype=pair.model.lbs.dtype)
    velocity = (torch.rand(size, **kwargs) - 0.5) * 2.0 * scale
    return coords, prev, velocity
