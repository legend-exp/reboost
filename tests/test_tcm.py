from __future__ import annotations

import awkward as ak
import lh5
import numpy as np
import pytest
from lgdo import Array, Table

from reboost import io, tcm, utils


@pytest.fixture(scope="module")
def hit_file(tmptestdir, remage_stp_file):
    """Hit tier file with the event fields of every table in the remage output.

    The germanium tables are written in two chunks to exercise appending.
    """
    outfile = tmptestdir / "hit.lh5"

    for uid, det in utils.get_remage_detector_uids(remage_stp_file).items():
        buffer_len = 100 if det.startswith("det") else 10_000
        for chunk in lh5.LH5Iterator(remage_stp_file, f"stp/{det}", buffer_len=buffer_len):
            io.write_hit_table_chunk(io.init_hit_table(chunk), f"hit/{det}", outfile, uid=uid)

    return outfile


def test_build_remage_tcm(hit_file, remage_stp_file):
    tcm.build_remage_tcm(hit_file, hit_file)

    assert "tcm" in lh5.ls(hit_file)

    # same settings as remage: the tcm must be identical
    built = lh5.read_as("tcm", hit_file, "ak")
    expected = lh5.read_as("tcm", remage_stp_file, "ak")

    assert len(built) == len(expected)
    assert ak.all(built.table_key == expected.table_key)
    assert ak.all(built.row_in_table == expected.row_in_table)


def test_build_remage_tcm_separate_file(tmptestdir, hit_file, remage_stp_file):
    outfile = tmptestdir / "tcm.lh5"
    tcm.build_remage_tcm([hit_file], outfile, coin_window_in_ns=0)

    assert lh5.ls(outfile) == ["tcm"]

    # a null time window still groups hits with the same t0
    built = lh5.read_as("tcm", outfile, "ak")
    assert len(built) >= len(lh5.read_as("tcm", remage_stp_file, "ak"))


def test_build_remage_tcm_time_units(tmptestdir):
    hit_file = tmptestdir / "hit_us.lh5"

    table = Table(size=2)
    table.add_field("t0", Array(np.array([1.0, 2.0]), attrs={"units": "us"}))
    table.add_field("evtid", Array(np.array([1, 2])))
    io.write_hit_table_chunk(table, "hit/det001", hit_file, uid=1)

    with pytest.raises(ValueError, match="nanoseconds"):
        tcm.build_remage_tcm(hit_file, tmptestdir / "tcm_us.lh5")
