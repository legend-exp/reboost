from __future__ import annotations

import lh5
import numpy as np
import pytest
from scipy.interpolate import RegularGridInterpolator

from reboost.hpge.utils import (
    HPGePulseShapeLibrary,
    HPGeRZField,
    load_hpge_pulse_shape_library,
    load_hpge_rz_field,
    make_hpge_pulse_shape_library,
    make_hpge_rz_field,
)
from reboost.units import ureg as u


def test_read_hpge_map(legendtestdata):
    dt_map = load_hpge_rz_field(
        legendtestdata["lh5/hpge-drift-time-maps.lh5"],
        "V99000A",
        "drift_time",
        out_of_bounds_val=0,
    )

    assert isinstance(dt_map, HPGeRZField)

    assert dt_map.r_units == u.m
    assert dt_map.z_units == u.m
    assert dt_map.φ_units == u.ns

    assert isinstance(dt_map.φ, RegularGridInterpolator)

    with pytest.raises(ValueError):
        dt_map.φ((0, -1))

    assert dt_map.φ((0, 0)) == 0
    assert dt_map.φ([(0, 0.01), (0.03, 0.03)]) == pytest.approx([135, 695])


def test_read_pulse_shape_library(test_pulse_shape_library):
    # check th reading works
    lib = load_hpge_pulse_shape_library(test_pulse_shape_library, "V01", "waveforms")
    assert isinstance(lib, HPGePulseShapeLibrary)

    assert np.shape(lib.waveforms) == (200, 200, 4001)


def test_make_hpge_rz_field(legendtestdata):
    filename = legendtestdata["lh5/hpge-drift-time-maps.lh5"]
    points = [(0, 0.01), (0.03, 0.03)]

    loaded = load_hpge_rz_field(filename, "V99000A", "drift_time", out_of_bounds_val=0)
    made = make_hpge_rz_field(lh5.read("V99000A", filename), "drift_time", out_of_bounds_val=0)

    assert isinstance(made, HPGeRZField)
    assert made.r_units == loaded.r_units
    assert made.z_units == loaded.z_units
    assert made.φ_units == loaded.φ_units
    assert made.ndim == loaded.ndim
    assert np.array_equal(made.values, loaded.values)
    assert made.φ(points) == pytest.approx(loaded.φ(points))


def test_make_hpge_rz_field_from_mapping(legendtestdata):
    # the grid does not have to be an LGDO Struct: a mapping is enough
    filename = legendtestdata["lh5/hpge-drift-time-maps.lh5"]
    data = lh5.read("V99000A", filename)
    fields = {k: data[k] for k in ("r", "z", "drift_time")}

    made = make_hpge_rz_field(fields, "drift_time", out_of_bounds_val=0)

    assert made.φ((0, 0)) == 0


def test_make_hpge_pulse_shape_library(test_pulse_shape_library):
    loaded = load_hpge_pulse_shape_library(test_pulse_shape_library, "V01", "waveforms")
    made = make_hpge_pulse_shape_library(lh5.read("V01", test_pulse_shape_library), "waveforms")

    assert isinstance(made, HPGePulseShapeLibrary)
    assert made.r_units == loaded.r_units
    assert made.z_units == loaded.z_units
    assert made.t_units == loaded.t_units
    assert np.array_equal(made.t, loaded.t)
    assert np.array_equal(made.waveforms, loaded.waveforms)


def test_make_hpge_pulse_shape_library_dtype(test_pulse_shape_library):
    made = make_hpge_pulse_shape_library(
        lh5.read("V01", test_pulse_shape_library), "waveforms", dtype=np.float32
    )

    assert made.waveforms.dtype == np.float32
