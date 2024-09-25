from __future__ import annotations

import logging
from typing import Callable, Literal

import numpy as np
import pandas as pd
import scipy.optimize
from lgdo import lh5
from lgdo.lh5 import LH5Iterator, LH5Store
from lgdo.types import Array, Histogram, Scalar
from numba import njit
from numpy.typing import NDArray

from .optmap import OpticalMap

log = logging.getLogger(__name__)


def get_channel_efficiency(rawid: int) -> float:  # noqa: ARG001
    # TODO: implement
    return 0.99


def _optmaps_for_channels(
    optmap_events: pd.DataFrame,
    settings,
    chfilter: tuple[str] | Literal["*"] = (),
):
    all_det_ids = [ch_id for ch_id in list(optmap_events.columns) if ch_id.isnumeric()]
    eff = np.array([get_channel_efficiency(int(ch_id)) for ch_id in all_det_ids])

    if chfilter != "*":
        optmap_det_ids = [det for det in all_det_ids if det in chfilter]
    else:
        optmap_det_ids = all_det_ids

    log.info("creating empty optmaps")
    optmap_count = len(optmap_det_ids) + 1
    optmaps = [
        OpticalMap("all" if i == 0 else optmap_det_ids[i], settings) for i in range(optmap_count)
    ]

    return all_det_ids, eff, optmaps, optmap_det_ids


@njit(cache=True)
def _compute_hit_maps(hitcounts, eff, rng, optmap_count):
    mask = np.zeros((hitcounts.shape[0], optmap_count), dtype=np.bool_)
    counts = hitcounts.sum(axis=1)
    for idx in range(hitcounts.shape[0]):
        if counts[idx] == 0:
            continue

        for ch_idx in range(hitcounts.shape[1]):
            c = rng.binomial(hitcounts[idx, ch_idx], eff[ch_idx])
            if c > 0:  # detected
                mask[idx, 0] = True
                mask[idx, ch_idx + 1] = True
    return mask


def _fill_hit_maps(optmaps: list[OpticalMap], loc, hitcounts: NDArray, eff: NDArray, rng):
    masks = _compute_hit_maps(hitcounts, eff, rng, len(optmaps))

    for i in range(len(optmaps)):
        locm = loc[masks[:, i]]
        optmaps[i].h_hits.fill(locm[:, 0], locm[:, 1], locm[:, 2])


def _count_multi_ph_detection(hitcounts) -> NDArray:
    hits_per_primary = hitcounts.sum(axis=1)
    bins = np.arange(0, hits_per_primary.max() + 1.5) - 0.5
    return np.histogram(hits_per_primary, bins)[0]


def _fit_multi_ph_detection(hits_per_primary) -> float:
    x = np.arange(0, len(hits_per_primary))
    popt, pcov = scipy.optimize.curve_fit(
        lambda x, p0, k: p0 * np.exp(-k * x), x[1:], hits_per_primary[1:]
    )
    best_fit_exponent = popt[1]

    log.info(
        "p(> 1 detected photon)/p(1 detected photon) = %f",
        sum(hits_per_primary[2:]) / hits_per_primary[1],
    )
    log.info(
        "p(> 1 detected photon)/p(<=1 detected photon) = %f",
        sum(hits_per_primary[2:]) / sum(hits_per_primary[0:2]),
    )

    return best_fit_exponent


def create_optical_maps(
    optmap_events_it: LH5Iterator,
    settings,
    chfilter=(),
    output_lh5_fn=None,
    after_save: Callable[[int, str, OpticalMap]] | None = None,
) -> None:
    """
    Parameters
    ----------
    optmap_events_it
        :class:`LH5Iterator` of a table with columns ``{x,y,z}loc`` and one column (with numeric
        header) for each SiPM channel.
    chfilter
        tuple of detector ids that will be included in the resulting optmap. Those have to match
        the column names in ``optmap_events_it``.
    """
    optmap_events = optmap_events_it.read(0)[0].view_as("pd")
    all_det_ids, eff, optmaps, optmap_det_ids = _optmaps_for_channels(
        optmap_events, settings, chfilter=chfilter
    )

    hits_per_primary = np.zeros(10, dtype=np.int64)
    hits_per_primary_len = 0
    rng = np.random.default_rng()
    for it_count, (events_lgdo, events_entry, event_n_rows) in enumerate(optmap_events_it):
        assert (it_count == 0) == (events_entry == 0)
        optmap_events = events_lgdo.view_as("pd").iloc[0:event_n_rows]
        hitcounts = optmap_events[all_det_ids].to_numpy()
        loc = optmap_events[["xloc", "yloc", "zloc"]].to_numpy()

        log.info("filling vertex histogram (%d)", it_count)
        optmaps[0].fill_vertex(loc)

        log.info("computing map (%d)", it_count)
        _fill_hit_maps(optmaps, loc, hitcounts, eff, rng)
        hpp = _count_multi_ph_detection(hitcounts)
        hits_per_primary_len = max(hits_per_primary_len, len(hpp))
        hits_per_primary[0 : len(hpp)] += hpp

    hits_per_primary = hits_per_primary[0:hits_per_primary_len]
    hits_per_primary_exponent = _fit_multi_ph_detection(hits_per_primary)

    # all maps share the same vertex histogram.
    for i in range(1, len(optmaps)):
        optmaps[i].h_vertex = optmaps[0].h_vertex

    log.info("computing probability and storing to %s", output_lh5_fn)
    for i in range(len(optmaps)):
        optmaps[i].create_probability()
        optmaps[i].check_histograms()
        group = "all" if i == 0 else "_" + optmap_det_ids[i - 1]
        if output_lh5_fn is not None:
            optmaps[i].write_lh5(lh5_file=output_lh5_fn, group=group)

        if after_save is not None:
            after_save(i, group, optmaps[i])

        optmaps[i] = None

    if output_lh5_fn is not None:
        lh5.write(Array(hits_per_primary), "_hitcounts", lh5_file=output_lh5_fn)
        lh5.write(Scalar(hits_per_primary_exponent), "_hitcounts_exp", lh5_file=output_lh5_fn)


def merge_optical_maps(map_l5_files: list[str], output_lh5_fn: str, settings) -> None:
    store = LH5Store(keep_open=True)

    # verify that we have the same maps in all files.
    all_det_ntuples = None
    for optmap_fn in map_l5_files:
        maps = lh5.ls(optmap_fn)
        det_ntuples = [m for m in maps if m not in ("_hitcounts", "_hitcounts_exp")]
        if all_det_ntuples is not None and det_ntuples != all_det_ntuples:
            msg = "available optical maps in input files differ"
            raise ValueError(msg)
        all_det_ntuples = det_ntuples

    def _edges_eq(e1: tuple[NDArray], e2: tuple[NDArray]):
        return len(e1) == len(e2) and all(np.all(x1 == x2) for x1, x2 in zip(e1, e2))

    # merge maps one-by-one.
    for d in all_det_ntuples:
        merged_map = OpticalMap(d, settings)
        merged_map.h_vertex = merged_map.prepare_hist()
        merged_nr_gen = merged_map.h_vertex.view()
        merged_nr_det = merged_map.h_hits.view()

        all_edges = None
        for optmap_fn in map_l5_files:
            nr_det = store.read(f"/{d}/nr_det", optmap_fn)[0]
            assert isinstance(nr_det, Histogram)
            nr_gen = store.read(f"/{d}/nr_gen", optmap_fn)[0]
            assert isinstance(nr_gen, Histogram)

            optmap_edges = tuple([b.edges for b in nr_det.binning])
            optmap_edges_gen = tuple([b.edges for b in nr_gen.binning])
            assert _edges_eq(optmap_edges, optmap_edges_gen)
            if all_edges is not None and not _edges_eq(optmap_edges, all_edges):
                msg = "edges of input optical maps differ"
                raise ValueError(msg)
            all_edges = optmap_edges

            # now that we validated that they are equal, add up the actual data (in counts).
            merged_nr_det += nr_det.weights.nda
            merged_nr_gen += nr_gen.weights.nda

        merged_map.create_probability()
        merged_map.check_histograms()
        merged_map.write_lh5(lh5_file=output_lh5_fn, group=d)

    # merge hitcounts.
    hits_per_primary = np.zeros(10, dtype=np.int64)
    hits_per_primary_len = 0
    for optmap_fn in map_l5_files:
        hitcounts = store.read("/_hitcounts", optmap_fn)[0]
        assert isinstance(hitcounts, Array)
        hits_per_primary[0 : len(hitcounts)] += hitcounts
        hits_per_primary_len = max(hits_per_primary_len, len(hitcounts))

    hits_per_primary = hits_per_primary[0:hits_per_primary_len]
    lh5.write(Array(hits_per_primary), "_hitcounts", lh5_file=output_lh5_fn)

    # re-calculate hotcounts exponent.
    hits_per_primary_exponent = _fit_multi_ph_detection(hits_per_primary)
    lh5.write(Scalar(hits_per_primary_exponent), "_hitcounts_exp", lh5_file=output_lh5_fn)
