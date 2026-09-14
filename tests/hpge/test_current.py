from __future__ import annotations

import awkward as ak
import numpy as np
import pytest
from lgdo import VectorOfVectors

import reboost.hpge.surface as surface_module
from reboost import units
from reboost.hpge import psd, surface
from reboost.hpge.utils import HPGePulseShapeLibrary, load_hpge_pulse_shape_library
from reboost.shape import cluster
from reboost.units import ureg as u


@pytest.fixture(scope="module")
def test_model():
    # test getting the model
    model, x = psd.get_current_template(
        -1000,
        3000,
        1.0,
        amax=1,
        mean_aoe=1,
        mu=0,
        sigma=100,
        tau=100,
        tail_fraction=0.65,
        high_tail_fraction=0.1,
        high_tau=10,
    )

    mu = -x[np.argmax(model)]

    # with fixed mu
    model, x = psd.get_current_template(
        -1000,
        3000,
        1.0,
        amax=1,
        mean_aoe=0.5,
        mu=mu,
        sigma=100,
        tau=100,
        tail_fraction=0.65,
        high_tail_fraction=0.1,
        high_tau=10,
    )

    return model, x


def test_get_template(test_pulse_shape_library, compare_numba_vs_python):
    lib = load_hpge_pulse_shape_library(test_pulse_shape_library, "V01", "waveforms")

    ri, zi = compare_numba_vs_python(psd._get_template_idx, 10.0, 10.0, lib.r, lib.z)

    assert len(lib.waveforms[ri][zi]) == 4001


def test_maximum_current(test_model, compare_numba_vs_python):
    model, x = test_model

    # directly compare JIT vs Python for @njit leaf functions in the call chain
    compare_numba_vs_python(psd._njit_erf, np.linspace(-2.0, 2.0, 5))
    compare_numba_vs_python(psd._interpolate_pulse_model, model, 100.0, float(x[0]), 1.0, 0.0)
    edep_1 = np.array([100.0, 500.0])
    dt_1 = np.array([400.0, 700.0])
    compare_numba_vs_python(psd._get_waveform_value, 100.0, edep_1, dt_1, model, float(x[0]), 1.0)

    edep = units.attach_units(ak.Array([[100.0, 300.0, 50.0], [10.0, 0.0, 100.0], [500.0]]), "keV")
    times = units.attach_units(ak.Array([[400, 500, 700], [800, 0, 1500], [700]]), "ns")

    curr = psd.maximum_current(edep, times, template=model, times=x)
    assert isinstance(curr, ak.Array)

    assert len(curr) == 3

    # should be close to 250 (could be some differences due to the discretisation)
    assert abs(curr[2] - 250) < 0.1

    # test other return modes
    max_t = psd.maximum_current(
        edep,
        times,
        template=model,
        times=x,
        return_mode="max_time",
    )

    assert isinstance(max_t, ak.Array)
    assert len(max_t) == 3

    # should be close to 250 (could be some differences due to the discretisation)
    assert abs(max_t[2] - 700) < 2

    energy = psd.maximum_current(
        edep,
        times,
        template=model,
        times=x,
        return_mode="energy",
    )

    assert isinstance(energy, ak.Array)
    assert len(energy) == 3

    # should be close to 250 (could be some differences due to the discretisation)
    assert abs(energy[2] - 500.0) < 2


def test_units(test_model, compare_numba_vs_python):
    model, x = test_model

    # directly compare JIT vs Python for the @njit current-pulse model
    compare_numba_vs_python(
        psd._current_pulse_model, x.astype(np.float64), 1.0, 0.0, 100.0, 0.65, 100.0, 0.1, 10.0
    )

    # standard units
    edep = units.attach_units(ak.Array([[100.0, 300.0, 50.0], [10.0, 0.0, 100.0], [500.0]]), "keV")
    times = units.attach_units(ak.Array([[400, 500, 700], [800, 0, 1500], [700]]), "ns")

    energy = psd.maximum_current(edep, times, template=model, times=x, return_mode="energy")
    unit = units.get_unit_str(energy)

    assert unit == "keV"

    # try for time
    time = psd.maximum_current(edep, times, template=model, times=x, return_mode="max_time")

    unit = units.get_unit_str(time)
    assert unit == "ns"

    # try with different units and check the result is the same
    times_us = units.attach_units(
        ak.Array([[0.400, 0.500, 0.700], [0.800, 0, 1.500], [0.700]]), "us"
    )

    max_time_us = psd.maximum_current(
        edep, times_us, template=model, times=x, return_mode="max_time"
    )

    unit = units.get_unit_str(max_time_us)
    assert unit == "ns"

    assert ak.all(time == max_time_us)


def test_with_cluster(test_model, compare_numba_vs_python):
    model, x = test_model

    edep = units.attach_units(ak.Array([[100.0, 300.0, 50.0], [10.0, 1.0, 100.0], [500.0]]), "keV")
    times = units.attach_units(ak.Array([[400, 410, 420], [800, 0, 1500], [700]]), "ns")
    xloc = units.attach_units(ak.Array([[1, 1.1, 1.2], [0, 50, 80], [100]]), "mm")
    dist = units.attach_units(ak.Array([[50, 40, 0.2], [300, 0.4, 0.2], [0.8]]), "mm")

    yloc = ak.full_like(xloc, 0.0)
    zloc = ak.full_like(xloc, 0.0)
    trackid = ak.full_like(xloc, 0)

    # directly compare JIT vs Python for the @njit cluster function
    local_idx = ak.flatten(ak.local_index(trackid)).to_numpy()
    tid = ak.flatten(trackid).to_numpy()
    pos = np.vstack(
        [
            ak.flatten(units.units_conv_ak(xloc, "mm")).to_numpy().astype(np.float64),
            ak.flatten(units.units_conv_ak(yloc, "mm")).to_numpy().astype(np.float64),
            ak.flatten(units.units_conv_ak(zloc, "mm")).to_numpy().astype(np.float64),
        ]
    ).T
    dist_np = ak.flatten(units.units_conv_ak(dist, "mm")).to_numpy()
    compare_numba_vs_python(
        cluster._cluster_by_distance_numba,
        local_idx,
        tid,
        pos,
        dist_to_surf=dist_np,
        surf_cut=0.0,
        threshold=1.0,
        threshold_surf=1.0,
    )

    clusters = cluster.cluster_by_step_length(
        trackid, xloc, yloc, zloc, dist, threshold_in_mm=1, threshold_surf_in_mm=1, surf_cut=0
    )
    cluster_edep = cluster.apply_cluster(clusters, edep)
    cluster_times = cluster.apply_cluster(clusters, times)

    e = ak.sum(cluster_edep, axis=-1)
    t = ak.sum(cluster_edep * cluster_times, axis=-1) / e
    curr = psd.maximum_current(e, t, template=model, times=x)

    assert isinstance(curr, ak.Array)
    assert len(curr) == 3

    # should be close to 250 (could be some differences due to the discretisation)
    assert abs(curr[0] - 225) < 0.1


def test_maximum_current_surface(test_model, compare_numba_vs_python):
    model, x = test_model

    # test for both input types
    for dtype in [np.float64, np.float32]:
        edep = units.attach_units(
            ak.Array(
                ak.values_astype(
                    ak.Array([[100.0, 300.0, 50.0], [10.0, 0.0, 100.0], [500.0]]), dtype
                ),
            ),
            "keV",
        )

        times = units.attach_units(
            ak.Array(ak.values_astype(ak.Array([[400, 500, 700], [800, 0, 1500], [700]]), dtype)),
            "ns",
        )

        dist = units.attach_units(
            ak.Array(ak.values_astype(ak.Array([[50, 40, 0.2], [300, 0.4, 0.2], [0.8]]), dtype)),
            "mm",
        )

        surface_models = surface.get_surface_library(1002, 10)

        assert np.shape(surface_models)[0] == 10000
        assert np.shape(surface_models)[1] == 100
        surface_templates = psd.make_convolved_surface_library(model, surface_models)
        surface_activeness = surface_models[:, -1]

        # directly compare JIT vs Python for the surface-specific @njit functions
        charge = np.zeros(100)
        charge[50] = 1.0
        compare_numba_vs_python(surface_module._advance_diffusion, charge.copy(), 0.29)
        compare_numba_vs_python(surface_module._compute_diffusion_impl, charge.copy(), 100, 0.29)

        edep_1 = np.array([100.0, 0.2])
        dt_1 = np.array([400.0, 500.0])
        dist_1 = np.array([50.0, 0.1])  # second step is inside FCCD (1002 um)
        compare_numba_vs_python(
            psd._get_waveform_value_surface,
            100.0,
            edep_1,
            dt_1,
            dist_1,
            model,
            surface_templates.T,
            surface_activeness,
            10.0,
            1002.0,
            float(x[0]),
            1.0,
        )

        curr_surf = psd.maximum_current(
            edep,
            times,
            dist,
            template=model,
            fccd_in_um=1002,
            templates_surface=surface_templates,
            activeness_surface=surface_activeness,
            times=x,
            return_mode="current",
        )

        curr_bulk = psd.maximum_current(
            edep,
            times,
            dist,
            template=model,
            times=x,
            return_mode="current",
        )
        # check shape

        assert len(curr_surf) == 3

        # surface effects reduce the current
        assert np.all(curr_surf < curr_bulk)


def test_maximum_current_library(test_pulse_shape_library, compare_numba_vs_python):
    lib = load_hpge_pulse_shape_library(test_pulse_shape_library, "V01", "waveforms")

    model = lib.waveforms[0][0]
    x = lib.t

    # directly compare JIT vs Python for the library-specific @njit functions
    compare_numba_vs_python(psd._get_template_idx, 20.0, 40.0, lib.r, lib.z)

    edep_1 = np.array([100.0, 300.0])
    dt_1 = np.array([400.0, 500.0])
    r_1 = np.array([20.0, 10.0])
    z_1 = np.array([40.0, 2.0])
    pulse_shape_library = (lib.waveforms, lib.r, lib.z)
    compare_numba_vs_python(
        psd._get_waveform_value_pulse_shape_library,
        100.0,
        edep_1,
        dt_1,
        r_1,
        z_1,
        pulse_shape_library,
        float(x[0]),
        1.0,
    )

    edep = VectorOfVectors(
        ak.Array([[100.0, 300.0, 50.0], [10.0, 0.0, 100.0], [500.0]]), attrs={"unit": "keV"}
    )
    times = VectorOfVectors(
        ak.Array([[400, 500, 700], [800, 0, 1500], [700]], attrs={"unit": "ns"})
    )
    r = VectorOfVectors(
        ak.Array([[20.0, 10.0, 5.0], [10.0, 1.0, 0.0], [70.0]]), attrs={"unit": "mm"}
    )
    z = VectorOfVectors(
        ak.Array([[40.0, 2.0, 25.0], [22.0, 4.0, 1.2], [20.0]]), attrs={"unit": "mm"}
    )

    curr = psd.maximum_current(edep, times, template=model, times=x)
    curr2 = psd.maximum_current(edep, times, r=r, z=z, template=lib, times=x)

    assert ak.all(curr == curr2)

    assert isinstance(curr, ak.Array)


def test_maximum_current_library_units_conversion(test_pulse_shape_library):
    lib = load_hpge_pulse_shape_library(test_pulse_shape_library, "V01", "waveforms")

    scale = 1.0 + 0.01 * np.add.outer(np.arange(len(lib.r)), np.arange(len(lib.z)))
    waveforms = lib.waveforms * scale[:, :, np.newaxis]

    lib_mm = HPGePulseShapeLibrary(
        waveforms=waveforms,
        r_units=lib.r_units,
        z_units=lib.z_units,
        t_units=lib.t_units,
        r=lib.r,
        z=lib.z,
        t=lib.t,
    )
    lib_scaled = HPGePulseShapeLibrary(
        waveforms=waveforms,
        r_units=units.ureg.cm,
        z_units=units.ureg.cm,
        t_units=units.ureg.us,
        r=lib.r / 10.0,
        z=lib.z / 10.0,
        t=lib.t / 1000.0,
    )

    edep = units.attach_units(ak.Array([[500.0]]), "keV")
    times = units.attach_units(ak.Array([[700.0]]), "ns")
    r = units.attach_units(ak.Array([[20.0]]), "mm")
    z = units.attach_units(ak.Array([[40.0]]), "mm")

    curr_mm = psd.maximum_current(edep, times, r=r, z=z, template=lib_mm, times=lib_mm.t)
    curr_scaled = psd.maximum_current(
        edep, times, r=r, z=z, template=lib_scaled, times=lib_scaled.t
    )

    assert curr_scaled[0] > 0
    assert np.isclose(curr_mm[0], curr_scaled[0])


def test_coarser_template_sampling(test_model):
    """A template sampled every dt ns gives the same current as one sampled every ns."""
    model, x = test_model
    edep = units.attach_units(ak.Array([[100.0, 300.0], [500.0], [50.0, 50.0]]), "keV")
    times = units.attach_units(ak.Array([[400.0, 500.0], [700.0], [900.0, 1500.0]]), "ns")

    fine = ak.to_numpy(psd.maximum_current(edep, times, template=model, times=x))

    for dt in (2, 4, 8):
        coarse = ak.to_numpy(psd.maximum_current(edep, times, template=model[::dt], times=x[::dt]))
        # the template is smooth on these scales, so the maximum barely moves
        assert np.allclose(fine, coarse, rtol=1e-2)


def test_coarser_library_sampling(test_model):
    """Same, for a pulse-shape library."""
    model, x = test_model
    r = z = np.linspace(0, 100, 20)
    edep = units.attach_units(ak.Array([[100.0, 300.0], [500.0]]), "keV")
    times = units.attach_units(ak.Array([[400.0, 500.0], [700.0]]), "ns")
    r_step = units.attach_units(ak.Array([[10.0, 20.0], [30.0]]), "mm")
    z_step = units.attach_units(ak.Array([[10.0, 20.0], [30.0]]), "mm")

    out = {}
    for dt in (1, 2, 8):
        waveforms = np.zeros((len(r), len(z), len(model[::dt])))
        waveforms[:, :] = model[::dt]
        lib = HPGePulseShapeLibrary(waveforms, u.mm, u.mm, u.ns, r, z, x[::dt])
        out[dt] = ak.to_numpy(
            psd.maximum_current(
                edep, times, r=r_step, z=z_step, template=lib, times=None, return_mode="current"
            )
        )
        assert np.all(out[dt] > 0)

    assert np.allclose(out[1], out[2], rtol=1e-2)
    assert np.allclose(out[1], out[8], rtol=1e-2)


def test_surface_corrections_need_1ns_sampling(test_model):
    """The surface response is tabulated every ns, the bulk template must match."""
    model, x = test_model
    edep = units.attach_units(ak.Array([[100.0, 300.0], [500.0]]), "keV")
    times = units.attach_units(ak.Array([[400.0, 500.0], [700.0]]), "ns")
    dist = units.attach_units(ak.Array([[50.0, 0.2], [0.4]]), "mm")

    surface_models = np.zeros((2, 10))
    surface_models[:, -1] = 1.0
    templates = psd.make_convolved_surface_library(model[::2], surface_models)

    with pytest.raises(ValueError, match="every 1 ns"):
        psd.maximum_current(
            edep,
            times,
            dist,
            template=model[::2],
            times=x[::2],
            fccd_in_um=1002,
            templates_surface=templates,
            activeness_surface=surface_models[:, -1],
        )


def test_template_after_the_peak_is_used(test_model):
    """The samples past the maximum must not be dropped, whatever the alignment.

    A template whose peak sits at its centre used to lose everything after the
    peak, so the maximum was found only when the scan happened to land on it.
    """
    model, _ = test_model
    peak = int(np.argmax(model))
    # centre the peak: t0 = -peak, i.e. the degenerate alignment
    half = min(peak, len(model) - peak - 1)
    centred = model[peak - half : peak + half + 1]
    x_centred = np.arange(-half, half + 1, dtype=float)

    energies = np.array([100.0, 300.0, 50.0])
    edep = units.attach_units(ak.Array([[e] for e in energies]), "keV")
    # drift times deliberately off the 20 ns coarse-scan lattice
    times = units.attach_units(ak.Array([[413.0], [927.0], [1531.0]]), "ns")

    curr = ak.to_numpy(psd.maximum_current(edep, times, template=centred, times=x_centred))

    # one step per hit: the maximum is the peak of the template, times the energy
    assert np.allclose(curr, energies * centred.max(), rtol=1e-3)


def test_non_uniform_time_axis_is_rejected(test_model):
    model, x = test_model
    edep = units.attach_units(ak.Array([[100.0]]), "keV")
    times = units.attach_units(ak.Array([[400.0]]), "ns")

    bumpy = np.concatenate([x[:100], x[100::2]])
    with pytest.raises(ValueError, match="uniformly spaced"):
        psd.maximum_current(edep, times, template=model[: len(bumpy)], times=bumpy)


def test_missing_maximum_is_reported(caplog):
    """A hit whose pulse never enters the scanned window is flagged, not silently zero."""
    # a template that is zero everywhere but 500 ns after the charge is collected,
    # so the scan around the drift time finds nothing
    spike = np.zeros(1001)
    spike[500] = 1.0
    x = np.arange(1001, dtype=float)

    edep = units.attach_units(ak.Array([[100.0]]), "keV")
    times = units.attach_units(ak.Array([[400.0]]), "ns")

    with caplog.at_level("WARNING", logger="reboost.hpge.psd"):
        curr = psd.maximum_current(edep, times, template=spike, times=x)

    assert ak.to_numpy(curr)[0] == 0
    assert "no pulse maximum found" in caplog.text


def test_drift_time_outside_the_template_span(test_model):
    """The scan follows the drift times, which may sit outside the template axis.

    The template covers -1000 to 3000 ns, the hits are collected much later.
    """
    model, x = test_model
    energies = np.array([100.0, 300.0])
    edep = units.attach_units(ak.Array([[e] for e in energies]), "keV")
    times = units.attach_units(ak.Array([[50_000.0], [123_456.0]]), "ns")

    curr = ak.to_numpy(psd.maximum_current(edep, times, template=model, times=x))
    assert np.allclose(curr, energies * model.max(), rtol=1e-3)


def test_empty_hits_are_not_reported(test_model, caplog):
    """A hit with no energy gives zero current, which is not worth a warning."""
    model, x = test_model
    edep = units.attach_units(ak.Array([[0.0], [100.0]]), "keV")
    times = units.attach_units(ak.Array([[400.0], [500.0]]), "ns")

    with caplog.at_level("WARNING", logger="reboost.hpge.psd"):
        curr = psd.maximum_current(edep, times, template=model, times=x)

    assert ak.to_numpy(curr)[0] == 0
    assert "no pulse maximum found" not in caplog.text
