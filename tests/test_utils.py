from __future__ import annotations

from lgdo.types import VectorOfVectors

from reboost import utils
from reboost.utils import get_table_names


def test_table_names():
    names = "['hit/det001','hit/det002']"

    tcm = VectorOfVectors([[]], attrs={"tables": names})

    table_names = get_table_names(tcm)
    assert table_names["det001"] == 0
    assert table_names["det002"] == 1


def test_get_rmg_detector_uids(remage_stp_file):
    out = utils.get_remage_detector_uids(remage_stp_file)
    assert isinstance(out, dict)
    assert out == {1: "scint1", 2: "scint2", 11: "det1", 12: "det2", 101: "optdet1", 102: "optdet2"}
