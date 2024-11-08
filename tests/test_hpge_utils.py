from __future__ import annotations

import json
from pathlib import Path

import awkward as ak
import pytest
import yaml

from reboost.hpge.utils import _merge_arrays, load_dict


def test_merge():
    ak_obj = ak.Array({"evtid": [1, 1, 1, 1, 2, 2, 3], "edep": [100, 50, 1000, 20, 100, 200, 10]})
    bufer_rows = ak.Array({"evtid": [1, 1], "edep": [60, 50]})

    # should only remove the last element
    merged_idx_0, buffer_0, mode = _merge_arrays(ak_obj, None, 0, 100, True)

    assert ak.all(merged_idx_0.evtid == [1, 1, 1, 1, 2, 2])
    assert ak.all(merged_idx_0.edep == [100, 50, 1000, 20, 100, 200])

    assert ak.all(buffer_0.evtid == [3])
    assert ak.all(buffer_0.edep == [10])

    # delete input file
    assert mode == "of"

    # if delete input is false it should be appended
    _, _, mode = _merge_arrays(ak_obj, None, 0, 100, False)
    assert mode == "append"

    # now if idx isn't 0 or the max_idx should add the buffer and remove the end

    merged_idx, buffer, mode = _merge_arrays(ak_obj, bufer_rows, 2, 100, True)

    assert ak.all(merged_idx.evtid == [1, 1, 1, 1, 1, 1, 2, 2])
    assert ak.all(merged_idx.edep == [60, 50, 100, 50, 1000, 20, 100, 200])

    assert ak.all(buffer.evtid == [3])
    assert ak.all(buffer.edep == [10])

    assert mode == "append"

    # now for the final index just adds the buffer

    merged_idx_end, buffer_end, mode = _merge_arrays(ak_obj, bufer_rows, 100, 100, True)

    assert ak.all(merged_idx_end.evtid == [1, 1, 1, 1, 1, 1, 2, 2, 3])
    assert ak.all(merged_idx_end.edep == [60, 50, 100, 50, 1000, 20, 100, 200, 10])

    assert buffer_end is None


@pytest.fixture
def file_fixture(tmp_path):
    # Create a simple YAML file
    data = {"det": 1}
    yaml_file = tmp_path / "data.yaml"
    with Path.open(yaml_file, "w") as yf:
        yaml.dump(data, yf)

    json_file = tmp_path / "data.json"
    with Path.open(json_file, "w") as jf:
        json.dump(data, jf)

    # Create a simple TXT file
    txt_file = tmp_path / "data.txt"
    with Path.open(txt_file, "w") as tf:
        tf.write("Some text.\n")

    # Return paths for the test functions
    return {"yaml_file": yaml_file, "json_file": json_file, "txt_file": txt_file}


def test_read(file_fixture):
    json_dict = load_dict(file_fixture["json_file"], None)
    assert json_dict["det"] == 1

    yaml_dict = load_dict(file_fixture["yaml_file"], None)
    assert yaml_dict["det"] == 1

    with pytest.raises(NotImplementedError):
        load_dict(file_fixture["txt_file"], None)
