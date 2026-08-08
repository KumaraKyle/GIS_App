"""RTI solver, against geometries with known ground truth.

These validate the *solver* -- that a known disturbance is reconstructed near
where it was put, and that the resolution metrics honestly describe how badly
the geometry constrains things. They say nothing about whether a human body
attenuates a 2.4GHz link the way the ellipse model assumes; only hardware can
answer that.

The most important test here asserts a refusal: with two links the engine must
decline to produce an image at all.
"""

from __future__ import annotations

import numpy as np
import pytest

from csisense.config import AppConfig, GeometryConfig, LinkConfig, NodeConfig
from csisense.rti.engine import RtiEngine
from csisense.rti.geometry import from_config
from csisense.rti.solver import TikhonovSolver, build_weight_matrix

ROOM = (5.0, 4.0)


def ring_geometry(n_nodes: int, voxel_m: float = 0.25) -> GeometryConfig:
    """Nodes evenly spaced around the room perimeter, fully meshed."""
    w, h = ROOM
    perimeter = [
        (
            w * 0.5 + w * 0.48 * np.cos(2 * np.pi * i / n_nodes),
            h * 0.5 + h * 0.48 * np.sin(2 * np.pi * i / n_nodes),
        )
        for i in range(n_nodes)
    ]
    nodes = [NodeConfig(id=f"n{i}", position=p) for i, p in enumerate(perimeter)]
    links = [
        LinkConfig(link_id=f"n{i}-n{j}", tx=f"n{i}", rx=f"n{j}")
        for i in range(n_nodes)
        for j in range(i + 1, n_nodes)
    ]
    return GeometryConfig(room_size_m=ROOM, voxel_m=voxel_m, nodes=nodes, links=links)


def config_for(geometry: GeometryConfig, **rti: object) -> AppConfig:
    config = AppConfig()
    config.geometry = geometry
    for key, value in rti.items():
        setattr(config.rti, key, value)
    return config


def simulate(geometry, solver: TikhonovSolver, xy, *, strength=1.0, noise=0.0, seed=0):
    """Forward-model a point disturbance at xy into link measurements."""
    centres = geometry.grid.centres
    idx = int(np.argmin(np.linalg.norm(centres - np.asarray(xy), axis=1)))
    x_true = np.zeros(geometry.grid.n_voxels)
    x_true[idx] = strength
    y = solver.W @ x_true
    if noise:
        y = y + np.random.default_rng(seed).normal(0, noise * max(y.max(), 1e-9), y.shape)
    return y, idx


# --- the refusal ----------------------------------------------------------


def test_two_links_refuses_to_image():
    """320 voxels from 2 measurements: 318 dimensions of any image would come
    from the smoothness prior rather than from the radios."""
    geometry = GeometryConfig(
        room_size_m=ROOM,
        voxel_m=0.25,
        nodes=[
            NodeConfig(id="a", position=(0.2, 0.2)),
            NodeConfig(id="b", position=(4.8, 3.8)),
            NodeConfig(id="c", position=(0.2, 3.8)),
        ],
        links=[
            LinkConfig(link_id="ab", tx="a", rx="b"),
            LinkConfig(link_id="ac", tx="a", rx="c"),
        ],
    )
    engine = RtiEngine(config_for(geometry))
    status = engine.status()

    assert status.enabled is False
    assert status.mode == "ellipses"
    assert "needs" in status.reason

    frame = engine.update(0, {"ab": 9.0, "ac": 1.0})
    assert frame.image is None, "produced an image it cannot justify"
    assert frame.per_link == {"ab": 9.0, "ac": 1.0}


def test_ellipse_mode_still_says_which_link_fired():
    """With one or two links this is the complete truth available: something is
    somewhere on this curve."""
    geometry = GeometryConfig(
        room_size_m=ROOM,
        voxel_m=0.25,
        nodes=[NodeConfig(id="a", position=(0.5, 2.0)), NodeConfig(id="b", position=(4.5, 2.0))],
        links=[LinkConfig(link_id="ab", tx="a", rx="b")],
    )
    engine = RtiEngine(config_for(geometry))

    ellipse = engine.link_ellipse("ab")
    assert ellipse is not None
    assert ellipse.shape[1] == 2
    # The curve must enclose both radios.
    assert ellipse[:, 0].min() < 0.5
    assert ellipse[:, 0].max() > 4.5


def test_adding_nodes_lifts_the_gate_without_a_code_change():
    """The promise that RTI sharpens as hardware is added.

    Node count to link count goes N(N-1)/2, so three nodes is still only three
    links and stays gated; four nodes reaches six and crosses the threshold.
    """
    assert RtiEngine(config_for(ring_geometry(3))).status().enabled is False  # 3 links
    assert RtiEngine(config_for(ring_geometry(4))).status().enabled is True  # 6 links
    assert RtiEngine(config_for(ring_geometry(6))).status().enabled is True  # 15 links


# --- reconstruction -------------------------------------------------------


@pytest.mark.parametrize("n_nodes", [6, 8])
def test_point_disturbance_is_localised(n_nodes):
    geometry_config = ring_geometry(n_nodes)
    geometry = from_config(geometry_config)
    engine = RtiEngine(config_for(geometry_config))
    solver = engine.solver
    assert solver is not None

    truth = (2.5, 2.0)  # room centre, best-covered region
    y, _ = simulate(geometry, solver, truth)
    frame = engine.update(0, dict(zip(geometry.link_ids, y, strict=True)))

    assert frame.peak_xy is not None
    error = float(np.hypot(frame.peak_xy[0] - truth[0], frame.peak_xy[1] - truth[1]))
    tolerance = 2 * solver.resolution.effective_resolution_m
    assert error < tolerance, (
        f"{n_nodes} nodes: peak {error:.2f}m from truth, tolerance {tolerance:.2f}m"
    )


def test_more_links_give_finer_resolution():
    """The claim the whole 'add nodes' argument rests on."""
    resolutions = []
    for n_nodes in (3, 5, 7):
        geometry_config = ring_geometry(n_nodes)
        solver = TikhonovSolver(from_config(geometry_config), config_for(geometry_config).rti)
        resolutions.append(solver.resolution.effective_resolution_m)

    assert resolutions[0] > resolutions[1] > resolutions[2], (
        f"resolution did not improve with node count: {resolutions}"
    )


def test_reconstruction_survives_measurement_noise():
    geometry_config = ring_geometry(8)
    geometry = from_config(geometry_config)
    engine = RtiEngine(config_for(geometry_config))
    solver = engine.solver
    assert solver is not None

    truth = (2.5, 2.0)
    y, _ = simulate(geometry, solver, truth, noise=0.1, seed=3)
    frame = engine.update(0, dict(zip(geometry.link_ids, y, strict=True)))

    assert frame.peak_xy is not None
    error = float(np.hypot(frame.peak_xy[0] - truth[0], frame.peak_xy[1] - truth[1]))
    assert error < 3 * solver.resolution.effective_resolution_m


def test_no_disturbance_gives_no_peak():
    geometry_config = ring_geometry(6)
    engine = RtiEngine(config_for(geometry_config))
    frame = engine.update(0, dict.fromkeys(from_config(geometry_config).link_ids, 0.0))
    assert frame.peak_xy is None


# --- weights and honesty metrics -----------------------------------------


def test_weights_peak_on_the_line_between_radios():
    """A voxel on the direct path adds no excess length, so it must be the most
    sensitive point of the link."""
    geometry_config = GeometryConfig(
        room_size_m=ROOM,
        voxel_m=0.1,
        nodes=[NodeConfig(id="a", position=(1.0, 2.0)), NodeConfig(id="b", position=(4.0, 2.0))],
        links=[LinkConfig(link_id="ab", tx="a", rx="b")],
    )
    geometry = from_config(geometry_config)
    W = build_weight_matrix(geometry, sigma_e_m=0.25)

    centres = geometry.grid.centres
    on_path = int(np.argmin(np.linalg.norm(centres - np.array([2.5, 2.0]), axis=1)))
    off_path = int(np.argmin(np.linalg.norm(centres - np.array([2.5, 0.3]), axis=1)))

    assert W[0, on_path] > 10 * W[0, off_path]


def test_confidence_is_higher_where_coverage_is_higher():
    geometry_config = ring_geometry(6)
    geometry = from_config(geometry_config)
    solver = TikhonovSolver(geometry, config_for(geometry_config).rti)
    res = solver.resolution

    centres = geometry.grid.centres
    centre_idx = int(np.argmin(np.linalg.norm(centres - np.array([2.5, 2.0]), axis=1)))
    corner_idx = int(np.argmin(np.linalg.norm(centres - np.array([0.15, 0.15]), axis=1)))

    assert res.coverage[centre_idx] > res.coverage[corner_idx]


def test_status_caption_states_the_limits_plainly():
    """The caption is permanent UI, not a tooltip -- someone reading the screen
    should know how much to trust it without asking."""
    engine = RtiEngine(config_for(ring_geometry(6)))
    caption = engine.status().caption()

    assert "links" in caption
    assert "resolution" in caption
    assert "coverage" in caption


def test_disabled_by_config():
    engine = RtiEngine(config_for(ring_geometry(6), enabled=False))
    assert engine.status().enabled is False
    assert engine.status().mode == "off"


def test_unknown_node_in_a_link_is_rejected():
    """A typo in geometry config would otherwise silently drop a link and
    quietly degrade the image."""
    with pytest.raises(ValueError, match="unknown node"):
        GeometryConfig(
            nodes=[NodeConfig(id="a", position=(0.0, 0.0))],
            links=[LinkConfig(link_id="ab", tx="a", rx="nope")],
        )
