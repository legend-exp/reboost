from __future__ import annotations

import awkward as ak
import numpy as np
import pytest

from reboost import units
from reboost.hpge.psd import HPGePulseShapeLibrary, waveform_from_pulse_shape_library


@pytest.fixture
def pulse_shape_library():
    waveforms = np.ones((40, 120, 100))

    return HPGePulseShapeLibrary(
        waveforms, "mm", "mm", "ns", np.linspace(0, 20, 40), np.linspace(0, 60, 120), np.arange(100)
    )


def test_get_waveforms(pulse_shape_library):
    r = units.attach_units(ak.Array([[10, 30], [20]]), "mm")
    z = units.attach_units(ak.Array([[50, 40], [10]]), "mm")
    edep = units.attach_units(ak.Array([[100, 200], [600]]), "keV")

    waveforms = waveform_from_pulse_shape_library(edep, r, z, pulse_shape_library)

    assert waveforms.to_numpy().shape == (2, 100)
    assert np.all(waveforms[0].to_numpy() == 300)
    assert np.all(waveforms[1].to_numpy() == 600)
