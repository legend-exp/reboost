from __future__ import annotations

from lgdo import lh5

from reboost.hpge.psd import do_cluster, dt_heuristic
from reboost.hpge.utils import ReadDTFile
from reboost.shape.group import group_by_time


def test_dt_heuristic():
    det_pos = [50, 0, -30]
    dt_file = ReadDTFile(
        "/root/reboost/tests/hpge/test_files/B99000A_drift_time_map.lh5", "drift_times", *det_pos
    )
    cluster_size_mm = 0.1

    data = lh5.read_as("stp/det001/", "test_files/internal_electron.lh5", "ak")
    grouped_data = group_by_time(data, evtid_name="evtid").view_as("ak")

    _non_clustered_dth = dt_heuristic(grouped_data, dt_file)

    cluster_data = do_cluster(grouped_data, cluster_size_mm, dt_file)

    _clustered_dth = dt_heuristic(cluster_data, dt_file)


if __name__ == "__main__":
    test_dt_heuristic()
