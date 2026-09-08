from __future__ import annotations

from reboost import utils


def test_get_rmg_detector_uids(remage_stp_file):
    out = utils.get_remage_detector_uids(remage_stp_file)
    assert isinstance(out, dict)
    assert out == {1: "scint1", 2: "scint2", 11: "det1", 12: "det2", 101: "optdet1", 102: "optdet2"}
