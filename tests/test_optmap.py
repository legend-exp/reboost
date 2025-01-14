from __future__ import annotations

import numpy as np
import pytest
from lgdo import Array, Table, lh5

from reboost.optmap.convolve import convolve
from reboost.optmap.create import create_optical_maps, merge_optical_maps
from reboost.optmap.evt import build_optmap_evt


@pytest.fixture
def tbl_hits(tmp_path):
    evt_count = 100
    rng = np.random.default_rng(1234)
    loc = rng.uniform(size=(evt_count, 3))
    evtids = np.arange(1, evt_count + 1)

    tbl_vertices = Table(
        {
            "evtid": Array(evtids),
            "xloc": Array(loc[:, 0]),
            "yloc": Array(loc[:, 1]),
            "zloc": Array(loc[:, 2]),
            "n_part": Array(np.ones(evt_count)),
            "time": Array(np.ones(evt_count)),
        }
    )

    mask = rng.uniform(size=evt_count) < 0.2
    hit_count = np.sum(mask)

    tbl_optical = Table(
        {
            "evtid": Array(evtids[mask]),
            "det_uid": Array(np.ones(hit_count, dtype=np.int_)),
            "wavelength": Array(rng.normal(loc=400, scale=30, size=hit_count)),
            "time": Array(2 * np.ones(hit_count)),
        }
    )

    hit_file = tmp_path / "hit.lh5"
    lh5.write(tbl_vertices, name="stp/vertices", lh5_file=hit_file, wo_mode="overwrite_file")
    lh5.write(tbl_optical, name="stp/optical", lh5_file=hit_file, wo_mode="overwrite")
    return hit_file


def test_optmap_evt(tbl_hits, tmp_path):
    evt_out_file = tmp_path / "evt-out.lh5"
    build_optmap_evt(
        str(tbl_hits),
        str(evt_out_file),
        detectors=("1", "002", "003"),
        buffer_len=20,  # note: shorter window sizes (e.g. 10) do not work.
    )


@pytest.fixture
def tbl_evt_fns(tmp_path) -> tuple[str]:
    evt_count = 100
    rng = np.random.default_rng(1234)
    loc = rng.uniform(size=(evt_count, 3))
    hits = rng.geometric(p=0.9, size=(evt_count, 3)) - 1

    tbl_evt = Table(
        {
            "xloc": Array(loc[:, 0]),
            "yloc": Array(loc[:, 1]),
            "zloc": Array(loc[:, 2]),
            "001": Array(hits[:, 0]),
            "002": Array(hits[:, 1]),
            "003": Array(hits[:, 2]),
        }
    )

    evt_file = tmp_path / "evt.lh5"
    lh5.write(tbl_evt, name="optmap_evt", lh5_file=evt_file, wo_mode="overwrite_file")
    return (str(evt_file),)


@pytest.mark.filterwarnings("ignore::scipy.optimize._optimize.OptimizeWarning")
def test_optmap_create(tbl_evt_fns):
    settings = {
        "range_in_m": [[0, 1], [0, 1], [0, 1]],
        "bins": [10, 10, 10],
    }

    # test creation only with the summary map.
    create_optical_maps(
        tbl_evt_fns,
        settings,
        chfilter=(),
        output_lh5_fn=None,
    )

    # test creation with all detectors.
    create_optical_maps(
        tbl_evt_fns,
        settings,
        chfilter=("001", "002", "003"),
        output_lh5_fn=None,
    )

    # test creation with some detectors.
    create_optical_maps(
        tbl_evt_fns,
        settings,
        chfilter=("001"),
        output_lh5_fn=None,
    )

    # test creation on multiple cores.
    create_optical_maps(
        tbl_evt_fns,
        settings,
        chfilter=("001", "002", "003"),
        output_lh5_fn=None,
        n_procs=2,
    )


@pytest.mark.filterwarnings("ignore::scipy.optimize._optimize.OptimizeWarning")
def test_optmap_merge(tbl_evt_fns, tmp_path):
    settings = {
        "range_in_m": [[0, 1], [0, 1], [0, 1]],
        "bins": [10, 10, 10],
    }

    map1_fn = str(tmp_path / "map1.lh5")
    create_optical_maps(
        tbl_evt_fns,
        settings,
        chfilter=("001", "002", "003"),
        output_lh5_fn=map1_fn,
    )
    map2_fn = str(tmp_path / "map2.lh5")
    create_optical_maps(
        tbl_evt_fns,
        settings,
        chfilter=("001", "002", "003"),
        output_lh5_fn=map2_fn,
    )

    # test in sequential mode.
    map_merged_fn = str(tmp_path / "map-merged.lh5")
    merge_optical_maps([map1_fn, map2_fn], map_merged_fn, settings)

    # also test on multiple cores.
    map_merged_fn = str(tmp_path / "map-merged-mp.lh5")
    merge_optical_maps([map1_fn, map2_fn], map_merged_fn, settings, n_procs=2)


@pytest.fixture
def tbl_edep(tmp_path):
    evt_count = 100
    rng = np.random.default_rng(1234)
    loc = rng.uniform(size=(evt_count, 6))

    evtids = np.arange(1, evt_count + 1)

    tbl_edep = Table(
        {
            "evtid": Array(evtids),
            "particle": Array(22 * np.ones(evt_count, dtype=np.int64)),
            "edep": Array(rng.normal(loc=200, scale=2, size=evt_count)),
            "xloc_pre": Array(loc[:, 0]),
            "yloc_pre": Array(loc[:, 1]),
            "zloc_pre": Array(loc[:, 2]),
            "xloc_post": Array(loc[:, 3]),
            "yloc_post": Array(loc[:, 4]),
            "zloc_post": Array(loc[:, 5]),
            "time": Array(np.ones(evt_count)),
            "v_pre": Array(60 * np.ones(evt_count)),
            "v_post": Array(58 * np.ones(evt_count)),
        }
    )

    evt_file = tmp_path / "edep.lh5"
    lh5.write(tbl_edep, name="/stp/x", lh5_file=evt_file, wo_mode="overwrite_file")
    return evt_file


@pytest.mark.filterwarnings("ignore::scipy.optimize._optimize.OptimizeWarning")
def test_optmap_convolve(tbl_evt_fns, tbl_edep, tmp_path):
    settings = {
        "range_in_m": [[0, 1], [0, 1], [0, 1]],
        "bins": [2, 2, 2],
    }

    map_fn = str(tmp_path / "map.lh5")
    create_optical_maps(
        tbl_evt_fns,
        settings,
        chfilter=("001"),
        output_lh5_fn=map_fn,
    )

    out_fn = str(tmp_path / "convolved.lh5")
    convolve(
        map_fn,
        str(tbl_edep),
        "/stp/x",
        material="lar",
        output_file=out_fn,
        buffer_len=10,
    )
