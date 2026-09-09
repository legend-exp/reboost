from __future__ import annotations

import awkward as ak
import numpy as np
import pytest

from reboost import units
from reboost.hpge import (
    HPGePulseShapeLibrary,
    HPGeRZField,
    drift_time_crystal_axes,
    load_hpge_drift_time_maps,
    load_hpge_pulse_shape_libraries,
)
from reboost.units import ureg as u


@pytest.fixture
def dt_maps(hpge_crystal_axes_file):
    return load_hpge_drift_time_maps(hpge_crystal_axes_file, "V01")


def _steps(radius, angles_deg, zloc):
    """Single-step hits at a fixed radius and height, at the given azimuths."""
    phi = np.deg2rad(angles_deg)
    return [
        units.attach_units(ak.Array(np.expand_dims(coord, axis=-1)), "mm")
        for coord in (radius * np.cos(phi), radius * np.sin(phi), np.full(len(phi), zloc))
    ]


def test_load_hpge_drift_time_maps(dt_maps):
    assert set(dt_maps.keys()) == {0, 45}

    for angle, dt_map in dt_maps.items():
        assert isinstance(dt_map, HPGeRZField)
        assert dt_map.r_units == u.mm
        assert dt_map.z_units == u.mm
        assert dt_map.φ_units == u.ns

        scale = 1 if angle == 0 else 2
        assert dt_map.φ((30, 20)) == pytest.approx(scale * 50)


def test_load_hpge_drift_time_maps_missing_angle(hpge_crystal_axes_file):
    with pytest.raises(KeyError):
        load_hpge_drift_time_maps(hpge_crystal_axes_file, "V01", angles=(0, 30))


def test_load_hpge_pulse_shape_libraries(hpge_crystal_axes_file):
    libs = load_hpge_pulse_shape_libraries(hpge_crystal_axes_file, "V01", angles=(0, 45))

    assert set(libs.keys()) == {0, 45}

    for angle, lib in libs.items():
        assert isinstance(lib, HPGePulseShapeLibrary)
        assert np.shape(lib.waveforms) == (11, 11, 5)
        assert np.all(lib.waveforms == (1 if angle == 0 else 2))
        assert lib.t == pytest.approx([0, 1, 2, 3, 4])

    # only the requested angles are read
    assert set(load_hpge_pulse_shape_libraries(hpge_crystal_axes_file, "V01")) == {0}


def test_load_hpge_pulse_shape_libraries_dtype(hpge_crystal_axes_file):
    libs = load_hpge_pulse_shape_libraries(hpge_crystal_axes_file, "V01", dtype=np.float32)
    assert libs[0].waveforms.dtype == np.float32


def test_drift_time_crystal_axes(dt_maps):
    # r + z = 50 mm, so the map at 0 degrees gives 50 ns and the one at 45 degrees 100 ns
    angles = [0, 45, 90, 135, 180, 22.5]
    xloc, yloc, zloc = _steps(30, angles, 20)

    dt = drift_time_crystal_axes(xloc, yloc, zloc, dt_maps)

    assert ak.parameters(dt)["units"] == "ns"
    assert ak.flatten(dt).to_numpy() == pytest.approx([50, 100, 50, 100, 50, 75])


def test_drift_time_crystal_axes_coord_offset(dt_maps):
    xloc, yloc, zloc = _steps(30, [0, 45], 20)

    shifted = drift_time_crystal_axes(
        units.attach_units(xloc + 10, "mm"),
        units.attach_units(yloc - 5, "mm"),
        zloc,
        dt_maps,
        coord_offset=(10, -5, 0) * u.mm,
    )

    assert ak.flatten(shifted).to_numpy() == pytest.approx([50, 100])


def test_drift_time_crystal_axes_missing_map(dt_maps):
    with pytest.raises(KeyError):
        drift_time_crystal_axes(*_steps(30, [0], 20), {0: dt_maps[0]})
