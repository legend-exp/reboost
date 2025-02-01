from __future__ import annotations

import time

from reboost.profile import ProfileDict


def test_profile():
    # create the object
    time_dict = ProfileDict()

    time_start = time.time()

    # create some fields
    time_dict.update_field("global_objects", time_start - 5)
    time_dict.update_field("processing_groups/germanium/read", time_start - 3)
    time_dict.update_field("processing_groups/germanium/detector_objects", time_start - 17)
    time_dict.update_field("processing_groups/germanium/write", time_start - 0.01)
    time_dict.update_field("processing_groups/germanium/operations/energy", time_start - 3)
    time_dict.update_field("processing_groups/germanium/operations/activeness", time_start - 42)
    time_dict.update_field("processing_groups/sipms/write", time_start - 12)

    # now update some fields
    time_dict.update_field("processing_groups/sipms/write", time_start - 7)

    assert round(time_dict["global_objects"], 1) == 5.0
    assert round(time_dict["processing_groups"]["germanium"]["read"], 1) == 3.0
    assert round(time_dict["processing_groups"]["germanium"]["detector_objects"], 1) == 17.0
    assert round(time_dict["processing_groups"]["germanium"]["write"], 1) == 0.0
    assert round(time_dict["processing_groups"]["germanium"]["operations"]["energy"], 1) == 3.0
    assert round(time_dict["processing_groups"]["germanium"]["operations"]["activeness"], 1) == 42.0
    assert round(time_dict["processing_groups"]["sipms"]["write"], 1) == 19.0
