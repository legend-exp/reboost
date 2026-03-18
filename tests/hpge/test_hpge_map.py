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


def test_read_pulse_shape_library_with_phi(test_pulse_shape_library_with_phi):
    """Test reading pulse shape library with phi-dependent fields."""
    lib = get_hpge_pulse_shape_library(test_pulse_shape_library_with_phi, "V01", "waveforms")
    assert isinstance(lib, HPGePulseShapeLibrary)

    # Should have 4D array (r, z, phi, time)
    assert np.shape(lib.waveforms) == (20, 20, 2, 100)

    # Check phi angles are correctly extracted
    assert lib.phi is not None
    assert len(lib.phi) == 2
    assert lib.phi[0] == 0.0
    assert lib.phi[1] == 45.0

    # Check that different phi have different values (should differ by factor of 2)
    ratio = lib.waveforms[:, :, 1, :] / lib.waveforms[:, :, 0, :]
    assert np.allclose(ratio, 2.0)
