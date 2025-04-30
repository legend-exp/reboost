from __future__ import annotations

from pathlib import Path

import pytest
from scipy.interpolate import RegularGridInterpolator

from reboost.hpge.utils import HPGeScalarRZField, get_hpge_scalar_rz_field
from reboost.utils import u


def test_read_hpge_map():
    dt_map = get_hpge_scalar_rz_field(
        f"{Path(__file__).parent}/simulation/drift-time-maps.lh5",
        "V99999A",
        "drift_time",
        out_of_bounds_val=0,
    )

    assert isinstance(dt_map, HPGeScalarRZField)

    assert dt_map.r_units == u["m"]
    assert dt_map.z_units == u["m"]
    assert dt_map.φ_units == u["ns"]

    assert isinstance(dt_map.φ, RegularGridInterpolator)

    with pytest.raises(ValueError):
        dt_map.φ((0, -1))

    assert dt_map.φ((0, 0)) == 0
    assert dt_map.φ([(0, 0.001), (0.005, 0.005)]) == pytest.approx([393.0189, 105.1679])
