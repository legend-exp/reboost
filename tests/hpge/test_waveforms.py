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


@pytest.fixture
def pulse_shape_library_with_phi():
    """Create a PSL with phi-dependent templates at 0 and 45 degrees."""
    # Create waveforms with shape (r, z, phi, time)
    # Use different values for 0 and 45 degrees to test selection
    waveforms = np.zeros((40, 120, 2, 100))
    waveforms[:, :, 0, :] = 1.0  # 0 degree templates
    waveforms[:, :, 1, :] = 2.0  # 45 degree templates

    return HPGePulseShapeLibrary(
        waveforms,
        "mm",
        "mm",
        "ns",
        np.linspace(0, 20, 40),
        np.linspace(0, 60, 120),
        np.arange(100),
        np.array([0.0, 45.0]),
    )


def test_get_waveforms(pulse_shape_library):
    r = units.attach_units(ak.Array([[10, 30], [20]]), "mm")
    z = units.attach_units(ak.Array([[50, 40], [10]]), "mm")
    edep = units.attach_units(ak.Array([[100, 200], [600]]), "keV")

    waveforms = waveform_from_pulse_shape_library(edep, r, z, pulse_shape_library)

    assert waveforms.to_numpy().shape == (2, 100)
    assert np.all(waveforms[0].to_numpy() == 300)
    assert np.all(waveforms[1].to_numpy() == 600)

    waveforms = waveform_from_pulse_shape_library(
        ak.Array([[]]), ak.Array([[]]), ak.Array([[]]), pulse_shape_library
    )

    assert waveforms.to_numpy().shape == (1, 100)
    assert np.all(waveforms[0].to_numpy() == 0)


def test_get_waveforms_with_phi(pulse_shape_library_with_phi):
    """Test waveform extraction with phi-dependent templates."""
    r = units.attach_units(ak.Array([[10, 30], [20]]), "mm")
    z = units.attach_units(ak.Array([[50, 40], [10]]), "mm")
    edep = units.attach_units(ak.Array([[100, 200], [600]]), "keV")

    # Test phi = 0 degrees (should use 0-degree template with value 1.0)
    phi = units.attach_units(ak.Array([[0, 0], [0]]), "deg")
    waveforms = waveform_from_pulse_shape_library(edep, r, z, pulse_shape_library_with_phi, phi)

    assert waveforms.to_numpy().shape == (2, 100)
    assert np.all(waveforms[0].to_numpy() == 300)  # 100*1 + 200*1
    assert np.all(waveforms[1].to_numpy() == 600)  # 600*1

    # Test phi = 45 degrees (should use 45-degree template with value 2.0)
    phi = units.attach_units(ak.Array([[45, 45], [45]]), "deg")
    waveforms = waveform_from_pulse_shape_library(edep, r, z, pulse_shape_library_with_phi, phi)

    assert waveforms.to_numpy().shape == (2, 100)
    assert np.all(waveforms[0].to_numpy() == 600)  # 100*2 + 200*2
    assert np.all(waveforms[1].to_numpy() == 1200)  # 600*2

    # Test phi = 90 degrees (should use 0-degree template due to 45-deg symmetry)
    phi = units.attach_units(ak.Array([[90, 90], [90]]), "deg")
    waveforms = waveform_from_pulse_shape_library(edep, r, z, pulse_shape_library_with_phi, phi)

    assert waveforms.to_numpy().shape == (2, 100)
    assert np.all(waveforms[0].to_numpy() == 300)  # 100*1 + 200*1

    # Test mixed phi values (10 deg -> 0 deg, 50 deg -> 45 deg)
    phi = units.attach_units(ak.Array([[10, 50], [20]]), "deg")
    waveforms = waveform_from_pulse_shape_library(edep, r, z, pulse_shape_library_with_phi, phi)

    assert waveforms.to_numpy().shape == (2, 100)
    assert np.all(waveforms[0].to_numpy() == 500)  # 100*1 + 200*2
    assert np.all(waveforms[1].to_numpy() == 600)  # 600*1 (20 deg is closer to 0 than 45)


def test_get_waveforms_without_phi_on_phi_library(pulse_shape_library_with_phi):
    """Test that calling without phi on phi-enabled library works.

    When phi is not provided for a phi-enabled library, the function uses the first phi
    angle in the library (index 0).
    """
    r = units.attach_units(ak.Array([[10, 30]]), "mm")
    z = units.attach_units(ak.Array([[50, 40]]), "mm")
    edep = units.attach_units(ak.Array([[100, 200]]), "keV")

    waveforms = waveform_from_pulse_shape_library(edep, r, z, pulse_shape_library_with_phi)

    assert waveforms.to_numpy().shape == (1, 100)
    # Default to first phi (0 degrees, value=1.0): 100*1 + 200*1 = 300
    assert np.all(waveforms[0].to_numpy() == 300)


def test_get_waveforms_mismatched_shapes(pulse_shape_library):
    """Test that mismatched input shapes raise a ValueError."""
    r = units.attach_units(ak.Array([[10, 30], [20]]), "mm")
    z = units.attach_units(ak.Array([[50, 40], [10]]), "mm")
    edep = units.attach_units(ak.Array([[100, 200, 300], [600]]), "keV")  # mismatched

    with pytest.raises(ValueError, match="same number of steps"):
        waveform_from_pulse_shape_library(edep, r, z, pulse_shape_library)


def test_get_waveforms_mismatched_phi_shape(pulse_shape_library_with_phi):
    """Test that mismatched phi shape raises a ValueError."""
    r = units.attach_units(ak.Array([[10, 30]]), "mm")
    z = units.attach_units(ak.Array([[50, 40]]), "mm")
    edep = units.attach_units(ak.Array([[100, 200]]), "keV")
    phi = units.attach_units(ak.Array([[0, 0, 0]]), "deg")  # mismatched

    with pytest.raises(ValueError, match="same number of steps"):
        waveform_from_pulse_shape_library(edep, r, z, pulse_shape_library_with_phi, phi)


def test_get_waveforms_zero_energy(pulse_shape_library):
    """Test that zero energy depositions produce a zero waveform."""
    r = units.attach_units(ak.Array([[10, 30]]), "mm")
    z = units.attach_units(ak.Array([[50, 40]]), "mm")
    edep = units.attach_units(ak.Array([[0, 0]]), "keV")

    waveforms = waveform_from_pulse_shape_library(edep, r, z, pulse_shape_library)

    assert waveforms.to_numpy().shape == (1, 100)
    assert np.all(waveforms[0].to_numpy() == 0)


def test_get_waveforms_single_hit(pulse_shape_library):
    """Test waveform extraction for a single event with a single hit."""
    r = units.attach_units(ak.Array([[10]]), "mm")
    z = units.attach_units(ak.Array([[50]]), "mm")
    edep = units.attach_units(ak.Array([[500]]), "keV")

    waveforms = waveform_from_pulse_shape_library(edep, r, z, pulse_shape_library)

    assert waveforms.to_numpy().shape == (1, 100)
    # Template value is 1 everywhere, so result is 500 * 1 = 500
    assert np.all(waveforms[0].to_numpy() == 500)


def test_get_waveforms_out_of_bounds(pulse_shape_library):
    """Test that r/z values outside the grid are clamped to the grid boundary."""
    # Use values far outside the grid (r_grid: 0-20 mm, z_grid: 0-60 mm)
    r = units.attach_units(ak.Array([[100, -5]]), "mm")
    z = units.attach_units(ak.Array([[200, -10]]), "mm")
    edep = units.attach_units(ak.Array([[100, 200]]), "keV")

    # Should not raise, and since templates are all 1, result is sum of energies
    waveforms = waveform_from_pulse_shape_library(edep, r, z, pulse_shape_library)

    assert waveforms.to_numpy().shape == (1, 100)
    assert np.all(waveforms[0].to_numpy() == 300)  # (100 + 200) * 1
