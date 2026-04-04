from __future__ import annotations

import awkward as ak
import numpy as np
import pytest

from reboost.daq import core, run_daq_non_sparse
from reboost.daq.utils import print_random_crash_msg

# Default parameters for _run_daq_non_sparse_impl (matching the public API defaults)
_TAU = 500.0  # µs
_NOISE = 5.0  # keV
_SLOPE_THR = 0.01  # keV/µs
_TRIG_THR = 25.0  # keV
_WF_LEN = 100.0  # µs
_TRIG_POS = 50.0  # µs


def _make_evt(energies, rawids):
    """Build a minimal simulated-event Awkward array accepted by run_daq_non_sparse."""
    return ak.Array(
        {
            "evtid": list(range(len(energies))),
            "geds_energy_active": energies,
            "geds_rawid_active": rawids,
        }
    )


def _make_impl_evt(evtids, energies, rawids, t0s):
    """Build event array with *t0* field for direct use by _run_daq_non_sparse_impl."""
    return ak.Array(
        {
            "evtid": evtids,
            "geds_energy_active": energies,
            "geds_rawid_active": rawids,
            "t0": t0s,
        }
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_evt():
    """Five events with two active channels (rawids 1000, 2000)."""
    energies = [[100.0, 50.0], [30.0], [200.0, 150.0], [5.0], [80.0]]
    rawids = [[1000, 2000], [1000], [1000, 2000], [2000], [1000]]
    return _make_evt(energies, rawids)


@pytest.fixture
def no_trigger_evt():
    """Three events whose energies are all below the default trigger_threshold (25 keV)."""
    energies = [[1.0, 2.0], [3.0], [4.0, 5.0]]
    rawids = [[1000, 2000], [1000], [1000, 2000]]
    return _make_evt(energies, rawids)


# ---------------------------------------------------------------------------
# Tests for run_daq_non_sparse
# ---------------------------------------------------------------------------


def test_run_daq_non_sparse_return_types(simple_evt):
    daq_data, channel_ids = run_daq_non_sparse(simple_evt, n_sim_events=100, source_activity=1.0)

    assert isinstance(daq_data, ak.Array)
    assert isinstance(channel_ids, list)


def test_run_daq_non_sparse_output_fields(simple_evt):
    daq_data, _ = run_daq_non_sparse(simple_evt, n_sim_events=100, source_activity=1.0)

    expected_fields = {
        "evtid",
        "timestamp",
        "has_trigger",
        "has_pre_pulse",
        "has_post_pulse",
        "has_slope",
    }
    assert set(daq_data.fields) == expected_fields


def test_run_daq_non_sparse_channel_ids(simple_evt):
    _, channel_ids = run_daq_non_sparse(simple_evt, n_sim_events=100, source_activity=1.0)

    # channel_ids must contain the unique raw IDs present in the input events
    assert set(channel_ids) == {1000, 2000}
    # must be sorted
    assert channel_ids == sorted(channel_ids)


def test_run_daq_non_sparse_output_shapes(simple_evt):
    daq_data, channel_ids = run_daq_non_sparse(simple_evt, n_sim_events=100, source_activity=1.0)

    n_records = len(daq_data)
    n_channels = len(channel_ids)

    # scalar fields per record
    assert len(daq_data.evtid) == n_records
    assert len(daq_data.timestamp) == n_records

    # boolean matrix fields: shape (n_records, n_channels)
    for field in ("has_trigger", "has_pre_pulse", "has_post_pulse", "has_slope"):
        arr = daq_data[field]
        assert arr.ndim == 2
        assert len(arr) == n_records
        assert ak.to_numpy(ak.num(arr, axis=1)).tolist() == [n_channels] * n_records


def test_run_daq_non_sparse_boolean_fields(simple_evt):
    daq_data, _ = run_daq_non_sparse(simple_evt, n_sim_events=100, source_activity=1.0)

    for field in ("has_trigger", "has_pre_pulse", "has_post_pulse", "has_slope"):
        flat = ak.to_numpy(ak.flatten(daq_data[field]))
        assert flat.dtype == np.dtype("bool"), f"{field} should be boolean"


def test_run_daq_non_sparse_evtid_and_timestamp_types(simple_evt):
    daq_data, _ = run_daq_non_sparse(simple_evt, n_sim_events=100, source_activity=1.0)

    evtids = ak.to_numpy(daq_data.evtid)
    timestamps = ak.to_numpy(daq_data.timestamp)

    assert np.issubdtype(evtids.dtype, np.integer)
    assert np.issubdtype(timestamps.dtype, np.floating)


def test_run_daq_non_sparse_triggers_present(simple_evt):
    """At least one trigger must be recorded when events exceed the threshold."""
    daq_data, _ = run_daq_non_sparse(simple_evt, n_sim_events=100, source_activity=1.0)

    assert len(daq_data) > 0
    assert ak.any(daq_data.has_trigger)


def test_run_daq_non_sparse_no_triggers(no_trigger_evt):
    """When no event exceeds the trigger_threshold the output table is empty."""
    daq_data, channel_ids = run_daq_non_sparse(
        no_trigger_evt, n_sim_events=100, source_activity=1.0
    )

    assert len(daq_data) == 0
    assert set(channel_ids) == {1000, 2000}


def test_run_daq_non_sparse_custom_threshold(simple_evt):
    """Raising the trigger_threshold to above all energies suppresses all records."""
    daq_data, _ = run_daq_non_sparse(
        simple_evt, n_sim_events=100, source_activity=1.0, trigger_threshold=1000.0
    )

    assert len(daq_data) == 0


def test_run_daq_non_sparse_single_channel():
    """Works correctly when only one channel is present in all events."""
    evt = _make_evt([[50.0], [80.0], [30.0]], [[999], [999], [999]])
    daq_data, channel_ids = run_daq_non_sparse(evt, n_sim_events=50, source_activity=2.0)

    assert channel_ids == [999]
    for field in ("has_trigger", "has_pre_pulse", "has_post_pulse", "has_slope"):
        assert ak.to_numpy(ak.num(daq_data[field], axis=1)).tolist() == [1] * len(daq_data)


# ---------------------------------------------------------------------------
# Tests for utils
# ---------------------------------------------------------------------------


def test_print_random_crash_msg(capsys):
    rng = np.random.default_rng(42)
    print_random_crash_msg(rng)
    captured = capsys.readouterr()
    assert len(captured.out.strip()) > 0


# ---------------------------------------------------------------------------
# Tests for _run_daq_non_sparse_impl (numba @njit) via compare_numba_vs_python
# ---------------------------------------------------------------------------


def test_impl_single_trigger(compare_numba_vs_python):
    """One event above the trigger threshold produces exactly one DAQ record."""
    evt = _make_impl_evt([0], [[50.0]], [[1000]], [100.0])
    chids = np.array([1000], dtype=np.int64)

    evtid, timestamp, has_trigger, has_pre_pulse, has_post_pulse, has_slope = (
        compare_numba_vs_python(
            core._run_daq_non_sparse_impl,
            evt,
            chids,
            _TAU,
            _NOISE,
            _SLOPE_THR,
            _TRIG_THR,
            _WF_LEN,
            _TRIG_POS,
        )
    )

    assert len(evtid) == 1
    assert evtid[0] == 0
    # timestamp is shifted back by trigger_position
    assert np.isclose(timestamp[0], 100.0 - _TRIG_POS)
    assert has_trigger[0, 0]
    assert not has_pre_pulse[0, 0]
    assert not has_post_pulse[0, 0]
    assert not has_slope[0, 0]


def test_impl_no_triggers(compare_numba_vs_python):
    """Events below the trigger threshold produce an empty output."""
    evt = _make_impl_evt([0, 1], [[10.0], [15.0]], [[1000], [1000]], [100.0, 500.0])
    chids = np.array([1000], dtype=np.int64)

    evtid, *_ = compare_numba_vs_python(
        core._run_daq_non_sparse_impl,
        evt,
        chids,
        _TAU,
        _NOISE,
        _SLOPE_THR,
        _TRIG_THR,
        _WF_LEN,
        _TRIG_POS,
    )

    assert len(evtid) == 0


def test_impl_post_pulse(compare_numba_vs_python):
    """An energy deposit in the post-trigger window sets has_post_pulse."""
    # event 0 triggers; event 1 arrives 30 µs later (< waveform_length - trigger_position = 50 µs)
    evt = _make_impl_evt([0, 1], [[50.0], [10.0]], [[1000], [1000]], [100.0, 130.0])
    chids = np.array([1000], dtype=np.int64)

    evtid, _timestamp, _has_trigger, _has_pre_pulse, has_post_pulse, _has_slope = (
        compare_numba_vs_python(
            core._run_daq_non_sparse_impl,
            evt,
            chids,
            _TAU,
            _NOISE,
            _SLOPE_THR,
            _TRIG_THR,
            _WF_LEN,
            _TRIG_POS,
        )
    )

    assert len(evtid) == 1
    assert has_post_pulse[0, 0]


def test_impl_dead_time(compare_numba_vs_python):
    """An event inside the dead-time window (after post-trigger, before next waveform) is skipped."""
    # event 0 triggers at t=100; event 1 at t=170 is in dead zone (50 < dt=70 < 100);
    # event 2 at t=400 is free to trigger again
    evt = _make_impl_evt(
        [0, 1, 2],
        [[50.0], [30.0], [50.0]],
        [[1000], [1000], [1000]],
        [100.0, 170.0, 400.0],
    )
    chids = np.array([1000], dtype=np.int64)

    evtid, *_ = compare_numba_vs_python(
        core._run_daq_non_sparse_impl,
        evt,
        chids,
        _TAU,
        _NOISE,
        _SLOPE_THR,
        _TRIG_THR,
        _WF_LEN,
        _TRIG_POS,
    )

    # event 1 falls in the dead zone and must be skipped
    assert len(evtid) == 2
    assert evtid[0] == 0
    assert evtid[1] == 2


def test_impl_pre_pulse(compare_numba_vs_python):
    """A sub-threshold deposit inside the pre-trigger window sets has_pre_pulse."""
    # event 0 (below trigger, above noise) at t=100;
    # event 1 triggers at t=130 → t0_start = 80 µs, so event 0 (80 < 100 < 130) is a pre-pulse
    evt = _make_impl_evt([0, 1], [[10.0], [50.0]], [[1000], [1000]], [100.0, 130.0])
    chids = np.array([1000], dtype=np.int64)

    evtid, _timestamp, _has_trigger, has_pre_pulse, _has_post_pulse, _has_slope = (
        compare_numba_vs_python(
            core._run_daq_non_sparse_impl,
            evt,
            chids,
            _TAU,
            _NOISE,
            _SLOPE_THR,
            _TRIG_THR,
            _WF_LEN,
            _TRIG_POS,
        )
    )

    assert len(evtid) == 1
    assert has_pre_pulse[0, 0]


def test_impl_baseline_slope(compare_numba_vs_python):
    """A large prior deposit leaves a measurable tail that sets has_slope."""
    # event 0 at t=0 with 1000 keV triggers; event 1 at t=1000 also triggers.
    # baseline slope for event 1:
    #   E/tau * exp(-(t0_start - t_prev)/tau) = 1000/500 * exp(-950/500) ≈ 0.30 > 0.01
    evt = _make_impl_evt([0, 1], [[1000.0], [50.0]], [[1000], [1000]], [0.0, 1000.0])
    chids = np.array([1000], dtype=np.int64)

    evtid, _timestamp, _has_trigger, _has_pre_pulse, _has_post_pulse, has_slope = (
        compare_numba_vs_python(
            core._run_daq_non_sparse_impl,
            evt,
            chids,
            _TAU,
            _NOISE,
            _SLOPE_THR,
            _TRIG_THR,
            _WF_LEN,
            _TRIG_POS,
        )
    )

    assert len(evtid) == 2
    # second record: large tail from first event produces a slope
    assert has_slope[1, 0]
    # first record: no prior history, no slope
    assert not has_slope[0, 0]


def test_impl_multi_channel_trigger(compare_numba_vs_python):
    """Only the channel(s) exceeding the trigger threshold get has_trigger=True."""
    # ch1000 deposits 30 keV (> 25); ch2000 deposits 10 keV (< 25)
    evt = _make_impl_evt([0], [[30.0, 10.0]], [[1000, 2000]], [100.0])
    chids = np.array([1000, 2000], dtype=np.int64)

    evtid, _timestamp, has_trigger, _has_pre_pulse, _has_post_pulse, _has_slope = (
        compare_numba_vs_python(
            core._run_daq_non_sparse_impl,
            evt,
            chids,
            _TAU,
            _NOISE,
            _SLOPE_THR,
            _TRIG_THR,
            _WF_LEN,
            _TRIG_POS,
        )
    )

    assert len(evtid) == 1
    assert has_trigger[0, 0]  # ch1000 triggered
    assert not has_trigger[0, 1]  # ch2000 did not trigger
