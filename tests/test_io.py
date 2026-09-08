from __future__ import annotations

import logging

import awkward as ak
import h5py
import lh5
import numpy as np
import pytest
from lgdo import Array, Table, VectorOfVectors

from reboost import io, utils


@pytest.fixture(scope="module")
def hit_file(tmptestdir):
    outfile = tmptestdir / "read_hit_field_by_tcm.lh5"

    det001 = Table(
        {
            "energy": Array(np.array([100, 200, 400, 300]), attrs={"units": "keV"}),
            "times": VectorOfVectors([[0.1], [0.2, 0.3], [0.4, 98], [2]]),
            "psd": Table({"aoe": Array(np.array([1.0, 2.0, 3.0, 4.0]))}),
        }
    )
    det002 = Table(
        {
            "energy": Array(np.array([10, 70, 0, 56, 400, 400]), attrs={"units": "keV"}),
            "times": VectorOfVectors([[12], [], [-0.4, 0.4], [89], [1], [2]]),
            "psd": Table({"aoe": Array(np.array([5.0, 6.0, 7.0, 8.0, 9.0, 10.0]))}),
        }
    )

    io.write_hit_table_chunk(det001, "hit/det001", outfile, uid=1)
    io.write_hit_table_chunk(det002, "hit/det002", outfile, uid=2)

    return det001.view_as("ak"), det002.view_as("ak"), outfile


@pytest.fixture(scope="module")
def tcm():
    return ak.Array(
        {
            "table_key": [[1], [1], [1, 2], [2], [2], [1, 2], [2], [], [2]],
            "row_in_table": [[0], [1], [2, 0], [1], [2], [3, 3], [4], [], [5]],
        }
    )


def test_read_hit_field_by_tcm(hit_file, tcm):
    det001, det002, file = hit_file

    energy = io.read_hit_field_by_tcm(tcm, file, "energy")
    assert "units" not in ak.parameters(energy)

    # same shape as the TCM
    assert len(energy) == len(tcm)
    assert ak.all(ak.num(energy, axis=-1) == ak.num(tcm.table_key, axis=-1))

    # the data itself, in the order of the TCM
    assert energy[0] == det001.energy[0]
    assert energy[1] == det001.energy[1]
    assert ak.all(energy[2] == [det001.energy[2], det002.energy[0]])
    assert ak.all(energy[5] == [det001.energy[3], det002.energy[3]])
    assert energy[8] == det002.energy[5]

    # explicit uids, lgdo TCM, nested field, units
    uids = utils.get_remage_detector_uids(file, lh5_table="hit")
    tcm_lgdo = Table(
        {
            "table_key": VectorOfVectors(tcm.table_key),
            "row_in_table": VectorOfVectors(tcm.row_in_table),
        }
    )
    aoe = io.read_hit_field_by_tcm(tcm_lgdo, file, "psd/aoe", uids, with_units=True)
    assert "units" not in ak.parameters(aoe)
    assert ak.all(ak.flatten(aoe) == [1, 2, 3, 5, 6, 7, 4, 8, 9, 10])

    energy = io.read_hit_field_by_tcm(tcm, file, "energy", uids, with_units=True)
    assert ak.parameters(energy)["units"] == "keV"

    # vector of vectors fields
    times = io.read_hit_field_by_tcm(tcm, file, "times")
    assert len(times) == len(tcm)
    assert ak.all(times[2] == [[0.4, 98], [12]])


def test_read_hit_field_by_tcm_edge_cases(hit_file, tcm):
    _, _, file = hit_file

    # a slice of the TCM, with non contiguous rows in det002
    part = tcm[[2, 4, 7, 8]]
    energy = io.read_hit_field_by_tcm(part, file, "energy")
    assert ak.all(ak.flatten(energy) == [400, 10, 0, 400])
    assert ak.num(energy, axis=-1).to_list() == [2, 1, 0, 1]

    # no hits at all
    energy = io.read_hit_field_by_tcm(tcm[[7]], file, "energy")
    assert energy.to_list() == [[]]

    # a detector missing from uids
    with pytest.raises(ValueError, match="no table"):
        io.read_hit_field_by_tcm(tcm, file, "energy", {1: "det001"})


def test_init_hit_table_scalar_t0():
    chunk = Table(size=3)
    chunk.add_field("t0", Array(np.array([100.0, 200.0, 300.0]), attrs={"units": "ns"}))
    chunk.add_field("evtid", Array(np.array([1, 2, 3])))
    chunk.add_field("edep", VectorOfVectors(ak.Array([[1.0], [2.0, 3.0], []])))

    result = io.init_hit_table(chunk)

    assert isinstance(result, Table)
    assert len(result) == 3
    assert list(result.keys()) == ["t0", "evtid"]
    assert result.t0.attrs["units"] == "ns"
    np.testing.assert_array_equal(result.t0.nda, [100.0, 200.0, 300.0])
    np.testing.assert_array_equal(result.evtid.nda, [1, 2, 3])


def test_init_hit_table_vector_time():
    chunk = Table(size=3)
    chunk.add_field(
        "time",
        VectorOfVectors(
            ak.Array([[100.0, 200.0], [300.0], [400.0, 500.0, 600.0]]),
            attrs={"units": "ns"},
        ),
    )
    chunk.add_field("evtid", VectorOfVectors(ak.Array([[1, 2], [3], [4, 5, 6]])))

    result = io.init_hit_table(chunk)

    assert len(result) == 3
    assert result.t0.attrs["units"] == "ns"
    # t0 is the time of the first step of each hit
    np.testing.assert_array_equal(result.t0.nda, [100.0, 300.0, 400.0])
    # evtid is the event identifier of the first step
    np.testing.assert_array_equal(result.evtid.nda, [1, 3, 4])


def test_init_hit_table_time_units(caplog):
    chunk = Table(size=2)
    chunk.add_field("t0", Array(np.array([1.0, 2.0]), attrs={"units": "us"}))
    chunk.add_field("evtid", Array(np.array([1, 2])))

    with pytest.raises(ValueError, match="nanoseconds"):
        io.init_hit_table(chunk)

    chunk = Table(size=2)
    chunk.add_field("t0", Array(np.array([1.0, 2.0])))
    chunk.add_field("evtid", Array(np.array([1, 2])))

    with caplog.at_level(logging.WARNING, logger="reboost.io"):
        io.init_hit_table(chunk)

    assert "assuming nanoseconds" in caplog.text


def _chunk(size=3):
    chunk = Table(size=size)
    chunk.add_field("t0", Array(np.arange(size, dtype=float), attrs={"units": "ns"}))
    chunk.add_field("evtid", Array(np.arange(size)))
    return chunk


def test_write_hit_table_chunk(tmptestdir):
    outfile = tmptestdir / "write_hit_table_chunk.lh5"
    chunk = _chunk()

    # first write: creates file, group, table and soft link
    io.write_hit_table_chunk(chunk, "hit/det001", outfile, uid=1)
    assert lh5.ls(outfile) == ["hit"]
    assert "hit/det001" in lh5.ls(outfile, "hit/")
    assert lh5.ls(outfile, "hit/__by_uid__/") == ["hit/__by_uid__/det001"]

    # second write: appends rows, no new link
    io.write_hit_table_chunk(chunk, "/hit/det001/", outfile, uid=1)
    result = lh5.read("hit/det001", outfile)
    assert len(result) == 6
    np.testing.assert_array_equal(result.evtid.nda, [0, 1, 2, 0, 1, 2])
    assert result.t0.attrs["units"] == "ns"

    # third write: a new detector with its own link
    io.write_hit_table_chunk(chunk, "hit/det002", outfile, uid=2)
    assert utils.get_remage_detector_uids(outfile, lh5_table="hit") == {1: "det001", 2: "det002"}

    # the links are not fields of the tier struct, as in the remage output
    with h5py.File(outfile) as f:
        assert f["hit"].attrs["datatype"] == "struct{det001,det002}"
        assert f["hit/__by_uid__"].attrs["datatype"] == "struct{det001,det002}"
        assert f["hit/__by_uid__"].get("det002", getlink=True).path == "/hit/det002"

    # reading the tier struct does not follow the links
    assert set(lh5.read("hit", outfile).keys()) == {"det001", "det002"}

    # a second tier group in the same file
    io.write_hit_table_chunk(chunk, "opt/det003", outfile, uid=3)
    assert lh5.ls(outfile) == ["hit", "opt"]
    assert utils.get_remage_detector_uids(outfile, lh5_table="opt") == {3: "det003"}

    # a uid can be registered once only
    with pytest.raises(ValueError, match="already registered"):
        io.write_hit_table_chunk(chunk, "hit/det004", outfile, uid=1)


def test_write_hit_table_chunk_no_link(tmptestdir):
    outfile = tmptestdir / "write_hit_table_chunk_no_link.lh5"

    io.write_hit_table_chunk(_chunk(), "hit/det001", outfile)
    io.write_hit_table_chunk(_chunk(), "hit/det001", outfile)

    assert lh5.ls(outfile, "hit/") == ["hit/det001"]
    assert len(lh5.read("hit/det001", outfile)) == 6

    with pytest.raises(ValueError, match="tier group"):
        io.write_hit_table_chunk(_chunk(), "det001", outfile)


def test_get_rows_in_event_range(remage_stp_file):
    tcm = lh5.read("tcm", remage_stp_file)

    for det, uid in utils.get_remage_detector_uids(remage_stp_file).items():
        n_rows = lh5.read_n_rows(f"stp/{uid}", remage_stp_file)

        # lgdo and awkward inputs
        assert io.get_rows_in_event_range(tcm, det, 0, len(tcm) - 1) == (0, n_rows)
        assert io.get_rows_in_event_range(tcm.view_as("ak"), det, 0, len(tcm) - 1) == (0, n_rows)

        # divide into groups
        groups = [[0, 10], [11, 40], [41, 101], [102, len(tcm) - 1]]
        n = 0
        i_prev = 0

        for group in groups:
            i_start, n_entries = io.get_rows_in_event_range(tcm, det, *group)
            if n_entries > 0:
                assert i_start == i_prev
                i_prev = i_start + n_entries
            n += n_entries

        assert n == n_rows


def test_get_rows_in_event_range_edge_cases(caplog):
    tcm = ak.Array(
        {
            "table_key": [[1, 2], [2], [1, 2], [2]],
            "row_in_table": [[0, 0], [1], [1, 2], [3]],
        }
    )

    assert io.get_rows_in_event_range(tcm, 1, 0, 3) == (0, 2)
    assert io.get_rows_in_event_range(tcm, 2, 1, 2) == (1, 2)
    assert io.get_rows_in_event_range(tcm, 1, 1, 1) == (0, 0)
    assert io.get_rows_in_event_range(tcm, 99, 0, 3) == (0, 0)

    with pytest.raises(ValueError, match="non-negative"):
        io.get_rows_in_event_range(tcm, 1, -1, 3)

    with pytest.raises(ValueError, match="smaller"):
        io.get_rows_in_event_range(tcm, 1, 2, 1)

    with caplog.at_level(logging.WARNING, logger="reboost.io"):
        assert io.get_rows_in_event_range(tcm, 1, 0, 4) == (0, 2)

    assert "truncating" in caplog.text

    tcm = ak.Array({"table_key": [[1], [1]], "row_in_table": [[0], [2]]})

    with pytest.raises(ValueError, match="contiguous"):
        io.get_rows_in_event_range(tcm, 1, 0, 1)
