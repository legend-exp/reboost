from __future__ import annotations

import numpy as np
import pytest
from scipy.interpolate import RegularGridInterpolator

from reboost.hpge.utils import (
    HPGePulseShapeLibrary,
    HPGeRZField,
    get_hpge_pulse_shape_library,
    get_hpge_rz_field,
)
from reboost.units import ureg as u


def test_read_hpge_map(legendtestdata):
    dt_map = get_hpge_rz_field(
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
    lib = get_hpge_pulse_shape_library(test_pulse_shape_library, "V01", "waveforms")
    assert isinstance(lib, HPGePulseShapeLibrary)

    assert np.shape(lib.waveforms) == (200, 200, 4001)
