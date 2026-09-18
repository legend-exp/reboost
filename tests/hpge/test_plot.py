from __future__ import annotations

import matplotlib as mpl
import numpy as np
import pytest

mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from reboost.hpge import plot_drift_time_maps, plot_psl_aoe_maps, plot_rz_maps
from reboost.hpge.plot import _psl_aoe_map, symmetrize
from reboost.hpge.utils import (
    HPGePulseShapeLibrary,
    load_hpge_drift_time_maps,
    load_hpge_pulse_shape_library,
)
from reboost.units import ureg as u


def _maps(n_r=5, n_z=7):
    r = np.linspace(0, 40, n_r)
    z = np.linspace(0, 60, n_z)
    field = np.outer(np.linspace(1, 2, n_r), np.linspace(1, 3, n_z))
    return {0: field, 45: 1.1 * field}, r, z


def test_symmetrize():
    field = np.arange(6).reshape(2, 3)  # (r, z)
    out = symmetrize(field)
    assert out.shape == (3, 4)  # (z, 2 * r)
    # the two halves mirror each other about the centre
    assert np.array_equal(out[:, :2], np.fliplr(out[:, 2:]))


def test_plot_rz_maps_panels():
    maps, r, z = _maps()

    fig, axes = plot_rz_maps(maps, r, z, label="x", title="V01")
    assert isinstance(fig, Figure)
    # one panel per crystal axis, plus their ratio
    assert len(axes) == 3
    assert axes[0].get_ylabel() == "z [mm]"
    assert "V01" in axes[0].get_title()

    # a single axis gives a single panel, and no ratio
    fig, axes = plot_rz_maps({0: maps[0]}, r, z)
    assert len(axes) == 1


def test_plot_rz_maps_into_given_axes():
    """The caller can own the figure, and so its size and layout."""
    maps, r, z = _maps()

    fig, given = plt.subplots(ncols=3, figsize=(14, 6), sharey=True)
    out_fig, out_axes = plot_rz_maps(maps, r, z, axes=given)

    assert out_fig is fig
    assert list(out_axes) == list(given)
    assert all(ax.get_images() for ax in given)
    assert tuple(fig.get_size_inches()) == (14, 6)


def test_plot_rz_maps_wrong_number_of_axes():
    maps, r, z = _maps()

    _, given = plt.subplots(ncols=2)
    with pytest.raises(ValueError, match="3 panels to draw"):
        plot_rz_maps(maps, r, z, axes=given)


def test_plot_rz_maps_empty():
    _, r, z = _maps()
    with pytest.raises(ValueError, match="no maps"):
        plot_rz_maps({}, r, z)


def test_plot_drift_time_maps(test_drift_time_map_file):
    maps = load_hpge_drift_time_maps(test_drift_time_map_file, "V01")

    fig, axes = plot_drift_time_maps(maps, title="V01")
    assert len(axes) == 3
    assert fig.axes[-1].get_ylabel() == "ratio"


def test_plot_drift_time_maps_ratio_scale(test_drift_time_map_file):
    maps = load_hpge_drift_time_maps(test_drift_time_map_file, "V01")

    _, axes = plot_drift_time_maps(maps, ratio_vmin=0.9, ratio_vmax=1.2)
    assert axes[-1].images[0].get_clim() == (0.9, 1.2)


def test_plot_drift_time_maps_without_grid(test_drift_time_map_file):
    """A field built by hand, without its grid, cannot be drawn."""
    maps = load_hpge_drift_time_maps(test_drift_time_map_file, "V01")
    bare = {0: maps[0]._replace(r=None, z=None, values=None)}

    with pytest.raises(ValueError, match="no gridded values"):
        plot_drift_time_maps(bare)


def test_psl_aoe_map_normalisation():
    # a library whose bulk pulses peak at 2, with a few larger ones
    waveforms = np.zeros((4, 5, 10))
    waveforms[..., 5] = 2.0
    waveforms[0, 0, 5] = 8.0
    waveforms[3, 4] = np.nan  # outside the detector
    lib = HPGePulseShapeLibrary(
        waveforms, u.mm, u.mm, u.ns, np.linspace(0, 3, 4), np.linspace(0, 4, 5), np.arange(10.0)
    )

    aoe = _psl_aoe_map(lib)
    assert np.isnan(aoe[3, 4])
    # the bulk sits at one, to the resolution of the histogram the mode comes from
    assert np.isclose(np.nanmedian(aoe), 1, rtol=1e-2)
    assert np.isclose(aoe[0, 0], 4, rtol=1e-2)

    raw = _psl_aoe_map(lib, normalise=False)
    assert np.isclose(np.nanmedian(raw), 2)


def test_plot_psl_aoe_maps(test_pulse_shape_library):
    lib = load_hpge_pulse_shape_library(test_pulse_shape_library, "V01", "waveforms")

    fig, axes = plot_psl_aoe_maps({0: lib}, title="V01")
    assert len(axes) == 1  # only one crystal axis: no ratio panel
    assert fig.axes[-1].get_ylabel() == "A/E"
