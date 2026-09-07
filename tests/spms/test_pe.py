from __future__ import annotations

import awkward as ak
import lh5
import numpy as np
import pytest

from reboost.optmap import convolve
from reboost.spms.pe import (
    _listoffset_chain,
    cluster_photoelectrons,
    corrected_photoelectrons,
    emitted_scintillation_photons,
    number_of_detected_photoelectrons,
    photoelectron_times,
    smear_photoelectrons,
)
from reboost.units import attach_units


def test_forced_trigger_correction():
    # check that with every data event empty it does nothing

    pe, uid = corrected_photoelectrons(
        ak.Array([[], [1], [2, 3]]), ak.Array([[], [0], [0, 1]]), ak.Array([[]]), ak.Array([[]])
    )

    assert ak.all(pe == [[], [1], [2, 3]])
    assert ak.all(uid == [[], [0], [0, 1]])

    # check adding a constant
    pe, uid = corrected_photoelectrons(
        ak.Array([[], [1], [2, 3]]), ak.Array([[], [0], [0, 1]]), ak.Array([[1]]), ak.Array([[0]])
    )

    assert ak.all(pe == [[1], [2], [3, 3]])
    assert ak.all(uid == [[0], [0], [0, 1]])

    # check sorting
    pe, uid = corrected_photoelectrons(
        ak.Array([[], [1], [2, 3]]),
        ak.Array([[], [0], [0, 1]]),
        ak.Array([[1, 3]]),
        ak.Array([[1, 0]]),
    )
    assert ak.all(pe == [[3, 1], [4, 1], [5, 4]])
    assert ak.all(uid == [[0, 1], [0, 1], [0, 1]])


def test_emitted_scintillation_photons(compare_numba_vs_python, monkeypatch):
    # directly compare JIT vs Python for the @njit particle-type helper
    compare_numba_vs_python(convolve._pdgid_to_particle, 22)

    edep = ak.Array([[1.0, 2.0], [3.0]])
    particle = ak.Array([[22, 22], [22]])

    out = emitted_scintillation_photons(edep, particle, "lar")

    assert ak.num(out).tolist() == ak.num(edep).tolist()
    assert ak.all(out >= 0)
    assert ak.all(ak.values_astype(out, int) == out)

    # exercise the Python (py_func) path for the main iteration kernel
    monkeypatch.setattr(
        convolve,
        "_iterate_stepwise_depositions_scintillate",
        convolve._iterate_stepwise_depositions_scintillate.py_func,
    )
    out_py = emitted_scintillation_photons(edep, particle, "lar")
    assert ak.num(out_py).tolist() == ak.num(edep).tolist()
    assert ak.all(out_py >= 0)
    assert ak.all(ak.values_astype(out_py, int) == out_py)


def test_number_of_detected_photoelectrons(mock_optmap_for_convolve, monkeypatch):
    xloc = ak.Array([[0.1, 0.2], [0.3]])
    yloc = ak.Array([[0.1, 0.2], [0.3]])
    zloc = ak.Array([[0.1, 0.2], [0.3]])
    num_scint_ph = ak.Array([[10, 20], [30]])

    out = number_of_detected_photoelectrons(
        xloc,
        yloc,
        zloc,
        num_scint_ph,
        mock_optmap_for_convolve,
        "all",
    )

    assert ak.num(out).tolist() == ak.num(num_scint_ph).tolist()
    assert ak.all(out >= 0)
    assert ak.all(ak.values_astype(out, int) == out)

    # exercise the Python (py_func) path for the main iteration kernel
    monkeypatch.setattr(
        convolve,
        "_iterate_stepwise_depositions_numdet",
        convolve._iterate_stepwise_depositions_numdet.py_func,
    )
    out_py = number_of_detected_photoelectrons(
        xloc,
        yloc,
        zloc,
        num_scint_ph,
        mock_optmap_for_convolve,
        "all",
    )
    assert ak.num(out_py).tolist() == ak.num(num_scint_ph).tolist()
    assert ak.all(out_py >= 0)
    assert ak.all(ak.values_astype(out_py, int) == out_py)


def test_number_of_detected_photoelectrons_max(mock_optmap_for_convolve, monkeypatch):
    xloc = ak.Array([[0.1, 0.2], [0.3]])
    yloc = ak.Array([[0.1, 0.2], [0.3]])
    zloc = ak.Array([[0.1, 0.2], [0.3]])
    num_scint_ph = ak.Array([[1, 1], [3000]])

    out, is_max = number_of_detected_photoelectrons(
        xloc,
        yloc,
        zloc,
        num_scint_ph,
        mock_optmap_for_convolve,
        "all",
        max_pes_per_hit=5,
    )

    assert is_max.tolist() == [False, True]
    assert ak.num(out).tolist() == ak.num(num_scint_ph).tolist()
    assert ak.all(out >= 0)
    assert ak.all(ak.values_astype(out, int) == out)

    num_scint_ph = ak.Array([[1, 10000], [1]])

    out, is_max = number_of_detected_photoelectrons(
        xloc,
        yloc,
        zloc,
        num_scint_ph,
        mock_optmap_for_convolve,
        "all",
        max_pes_per_hit=5,
    )

    assert is_max.tolist() == [True, False]

    # exercise the Python (py_func) path for the main iteration kernel
    monkeypatch.setattr(
        convolve,
        "_iterate_stepwise_depositions_numdet",
        convolve._iterate_stepwise_depositions_numdet.py_func,
    )
    num_scint_ph = ak.Array([[1, 1], [3000]])
    _out_py, is_max_py = number_of_detected_photoelectrons(
        xloc,
        yloc,
        zloc,
        num_scint_ph,
        mock_optmap_for_convolve,
        "all",
        max_pes_per_hit=5,
    )
    assert is_max_py.tolist() == [False, True]


def test_photoelectron_times(compare_numba_vs_python, monkeypatch):
    # directly compare JIT vs Python for the @njit particle-type helper
    compare_numba_vs_python(convolve._pdgid_to_particle, 22)

    num_det_ph = ak.Array([[0, 2], [1]])
    particle = ak.Array([[22, 22], [22]])
    time = ak.Array([[0.0, 1.0], [2.0]])

    out = photoelectron_times(num_det_ph, particle, time, "lar")

    assert ak.num(out).tolist() == ak.sum(num_det_ph, axis=1).tolist()
    assert ak.all(out >= 0)

    # exercise the Python (py_func) path for the main iteration kernel
    monkeypatch.setattr(
        convolve,
        "_iterate_stepwise_depositions_times",
        convolve._iterate_stepwise_depositions_times.py_func,
    )
    out_py = photoelectron_times(num_det_ph, particle, time, "lar")
    assert ak.num(out_py).tolist() == ak.sum(num_det_ph, axis=1).tolist()
    assert ak.all(out_py >= 0)


def test_cluster_photoelectrons_does_not_cross_subarrays():
    """Test that clustering does not merge elements across subarray boundaries."""
    times = ak.Array([[[0.0, 0.6], [0.7, 0.9]]])
    amps = ak.Array([[[1.0, 2.0], [3.0, 4.0]]])

    t_out, a_out = cluster_photoelectrons(times, amps, thr=1.0)

    assert ak.to_list(t_out) == [[[0.0], [0.7]]]
    assert ak.to_list(a_out) == [[[3.0], [7.0]]]


def test_cluster_photoelectrons_enforces_max_span():
    """Test that clusters respect the maximum time span threshold."""
    times = ak.Array([[0.0, 0.6, 1.1, 1.4, 2.3]])
    amps = ak.Array([[1.0, 2.0, 3.0, 4.0, 5.0]])

    t_out, a_out = cluster_photoelectrons(times, amps, thr=1.0)

    assert ak.to_list(t_out) == [[0.0, 1.1, 2.3]]
    assert ak.to_list(a_out) == [[3.0, 7.0, 5.0]]


def test_cluster_photoelectrons_empty_and_boundary():
    """Test clustering with empty arrays and exact boundary conditions."""
    times = ak.Array([[], [0.0, 1.0, 1.0001]])
    amps = ak.Array([[], [1.0, 2.0, 3.0]])

    t_out, a_out = cluster_photoelectrons(times, amps, thr=1.0)

    # [0.0, 1.0] spans exactly 1.0 -> same cluster; 1.0001 starts new
    assert ak.to_list(t_out) == [[], [0.0, 1.0001]]
    assert ak.to_list(a_out) == [[], [3.0, 3.0]]


def test_cluster_photoelectrons_mismatched_shapes():
    """Test that mismatched array shapes raise ValueError."""
    # different nesting depths
    times_1d = ak.Array([0.0, 1.0, 2.0])
    amps_2d = ak.Array([[1.0, 2.0, 3.0]])

    with pytest.raises(ValueError, match="nesting depth"):
        cluster_photoelectrons(times_1d, amps_2d, thr=1.0)

    # same nesting but different list lengths
    times = ak.Array([[0.0, 1.0], [2.0]])
    amps = ak.Array([[1.0], [2.0, 3.0]])

    with pytest.raises(ValueError, match="mismatched list lengths"):
        cluster_photoelectrons(times, amps, thr=1.0)


def test_cluster_photoelectrons_units(remage_stp_file):
    """Test clustering of realistic optical detector hit times."""
    times = lh5.read_as("stp/optdet1/time", remage_stp_file, "ak")
    times = ak.sort(times, axis=-1)
    amps = ak.ones_like(times)
    times = attach_units(times, "us")

    t_out, a_out = cluster_photoelectrons(times, amps, thr=10)

    # units are converted to ns before clustering and re-attached on output
    assert ak.parameters(t_out)["units"] == "ns"
    assert ak.all(ak.num(t_out) <= ak.num(times))
    assert ak.sum(a_out) == ak.sum(amps)

    # with a threshold below the time granularity nothing is merged
    t_out, _ = cluster_photoelectrons(times, amps, thr=0)
    assert ak.all(ak.num(t_out) == ak.num(times))


def test_smear_photoelectrons_shape_preservation():
    """Test that smear_photoelectrons preserves input array shape for 1D ragged arrays."""
    # test with 1D ragged arrays (the intended use case)
    array_1d = ak.Array([[1.0, 2.0, 3.0], [4.0], [5.0, 6.0]])
    array_empty = ak.Array([[], [1.0, 2.0], []])

    rng = np.random.default_rng(42)
    result_1d = smear_photoelectrons(array_1d, fwhm_in_pe=0.5, rng=rng)
    # check that the structure is preserved (number of elements per sublist)
    assert ak.num(result_1d, axis=1).to_list() == ak.num(array_1d, axis=1).to_list()
    assert len(result_1d) == len(array_1d)

    rng = np.random.default_rng(42)
    result_empty = smear_photoelectrons(array_empty, fwhm_in_pe=0.5, rng=rng)
    assert ak.num(result_empty, axis=1).to_list() == ak.num(array_empty, axis=1).to_list()
    assert len(result_empty) == len(array_empty)


def test_smear_photoelectrons_non_negativity():
    """Test that smear_photoelectrons clamps negative values to zero."""
    # use large FWHM to increase probability of negative samples. with loc=1
    # and sigma=2/2.35482≈0.85, ~12% of samples would be negative
    array = ak.Array([np.ones(10000)])
    rng = np.random.default_rng(42)
    result = smear_photoelectrons(array, fwhm_in_pe=2.0, rng=rng)

    flat_result = ak.flatten(result, axis=None)
    assert ak.all(flat_result >= 0)

    # verify that clamping actually occurred (some zeros should exist)
    assert ak.sum(flat_result == 0) > 0


def test_smear_photoelectrons_statistical_properties():
    """Test that smear_photoelectrons produces correct statistical distribution."""
    n_samples = 100000
    array = ak.Array([np.ones(n_samples)])
    fwhm = 0.4
    expected_sigma = fwhm / 2.35482

    rng = np.random.default_rng(42)
    result = smear_photoelectrons(array, fwhm_in_pe=fwhm, rng=rng)

    flat_result = ak.flatten(result, axis=None)
    # filter out clamped zeros for statistical analysis of the untruncated
    # distribution
    non_zero = flat_result[flat_result > 0]

    # mean should be close to 1 (allowing for small statistical fluctuation).
    # with small FWHM, very few values get clamped, so the mean is about 1
    assert 0.99 < ak.mean(non_zero) < 1.01

    # standard deviation should match expected value (with tolerance).
    # excluding clamped values gives us the true Gaussian sigma
    std = ak.std(non_zero)
    assert 0.95 * expected_sigma < std < 1.05 * expected_sigma


def test_smear_photoelectrons_reproducibility():
    """Test that smear_photoelectrons is reproducible with same seed."""
    array = ak.Array([[1.0, 2.0, 3.0, 4.0]])
    fwhm = 0.5

    rng1 = np.random.default_rng(123)
    result1 = smear_photoelectrons(array, fwhm_in_pe=fwhm, rng=rng1)

    rng2 = np.random.default_rng(123)
    result2 = smear_photoelectrons(array, fwhm_in_pe=fwhm, rng=rng2)

    assert ak.all(result1 == result2)


def test_smear_photoelectrons_default_rng():
    """Test that smear_photoelectrons works with the default RNG."""
    array = ak.Array([[1.0, 2.0, 3.0]])

    result = smear_photoelectrons(array, fwhm_in_pe=0.5)

    assert ak.num(result, axis=1).to_list() == ak.num(array, axis=1).to_list()
    assert ak.all(ak.flatten(result, axis=None) >= 0)


def test_listoffset_chain_1d():
    """Test _listoffset_chain with a 1D list-of-values array."""
    offsets_chain, content = _listoffset_chain(ak.to_layout(ak.Array([[1.0, 2.0], [3.0]])))

    assert len(offsets_chain) == 1
    np.testing.assert_array_equal(offsets_chain[0], [0, 2, 3])
    assert isinstance(content, ak.contents.NumpyArray)


def test_listoffset_chain_nested():
    """Test _listoffset_chain with a doubly-nested array (3D)."""
    arr = ak.Array([[[1.0, 2.0], [3.0]], [[4.0, 5.0, 6.0]]])
    offsets_chain, content = _listoffset_chain(ak.to_layout(arr))

    assert len(offsets_chain) == 2
    assert isinstance(content, ak.contents.NumpyArray)


def test_listoffset_chain_non_numpy_content():
    """Test _listoffset_chain raises TypeError when content is not NumpyArray."""
    # a list of records does not end in a NumpyArray
    arr = ak.Array([{"x": 1}, {"x": 2}])

    with pytest.raises(TypeError, match="NumpyArray"):
        _listoffset_chain(ak.to_layout(arr))
