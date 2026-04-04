from __future__ import annotations

import awkward as ak
import numpy as np
import pytest

from reboost.daq import run_daq_non_sparse
from reboost.daq.utils import print_random_crash_msg


def _make_evt(energies, rawids):
    """Build a minimal simulated-event Awkward array accepted by run_daq_non_sparse."""
    return ak.Array(
        {
            "evtid": list(range(len(energies))),
            "geds_energy_active": energies,
            "geds_rawid_active": rawids,
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
