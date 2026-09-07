from __future__ import annotations

import awkward as ak
import lh5
import pytest
from lgdo import Array, Struct, Table, VectorOfVectors

from reboost import core


@pytest.fixture(scope="module")
def hitfiles(tmptestdir):
    # make some hit tier files
    channel1 = Table(
        {
            "energy": Array([100, 200, 400, 300], attrs={"units": "keV"}),
            "times": VectorOfVectors([[0.1], [0.2, 0.3], [0.4, 98], [2]]),
        }
    )
    channel2 = Table(
        {
            "energy": Array([10, 70, 0, 56, 400, 400], attrs={"units": "keV"}),
            "times": VectorOfVectors([[12], [], [-0.4, 0.4], [89], [1], [2]]),
        }
    )

    lh5.write(Struct({"det001": channel1}), "hit", f"{tmptestdir}/hit_file_test.lh5", wo_mode="of")
    lh5.write(
        Struct({"det002": channel2}),
        "hit",
        f"{tmptestdir}/hit_file_test.lh5",
        wo_mode="append_column",
    )

    return channel1.view_as("ak"), channel2.view_as("ak"), f"{tmptestdir}/hit_file_test.lh5"


def test_read_data_at_channel(hitfiles):
    # make a TCM
    tcm_channels = ak.Array([[0], [0], [0, 1], [1], [1], [0, 1], [1], [], [1]])
    tcm_rows = ak.Array([[0], [1], [2, 0], [1], [2], [3, 3], [4], [], [5]])

    energy = core.read_data_at_channel_as_ak(
        tcm_channels, tcm_rows, hitfiles[2], "energy", "hit", {"det001": 0, "det002": 1}
    )
    assert "units" not in ak.parameters(energy)

    # check the same
    assert len(energy) == len(tcm_channels)
    assert ak.all(ak.num(energy, axis=-1) == ak.num(tcm_channels, axis=-1))

    # check the data itself
    assert energy[0] == hitfiles[0].energy[0]
    assert energy[1] == hitfiles[0].energy[1]
    assert ak.all(energy[2] == [hitfiles[0].energy[2], hitfiles[1].energy[0]])

    # also check for VoV
    times = core.read_data_at_channel_as_ak(
        tcm_channels, tcm_rows, hitfiles[2], "times", "hit", {"det001": 0, "det002": 1}
    )
    assert len(times) == len(tcm_channels)

    energy = core.read_data_at_channel_as_ak(
        tcm_channels,
        tcm_rows,
        hitfiles[2],
        "energy",
        "hit",
        {"det001": 0, "det002": 1},
        with_units=True,
    )
    assert "units" in ak.parameters(energy)
    assert ak.parameters(energy)["units"] == "keV"
