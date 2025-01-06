from __future__ import annotations

import logging
import multiprocessing as mp
from typing import Callable, Literal

import numpy as np
import scipy.optimize
from lgdo import Array, Histogram, Scalar, Struct, lh5
from lgdo.lh5 import LH5Store
from numba import njit
from numpy.typing import NDArray

from ..log_utils import setup_log
from .evt import EVT_TABLE_NAME, read_optmap_evt
from .optmap import OpticalMap

log = logging.getLogger(__name__)


# This is only a hotfix to ensure that histogram axes are pickleable... urgh.
def _struct_update_datatype(self) -> None:
    if not hasattr(self, "attrs"):
        return
    self.attrs["datatype"] = self.form_datatype()


if mp.current_process() != "MainProcess":
    Struct.update_datatype = _struct_update_datatype


def get_channel_efficiency(rawid: int, settings) -> float:  # noqa: ARG001
    # TODO: implement
    return 0.99


def _optmaps_for_channels(
    optmap_evt_columns: list[str],
    settings,
    chfilter: tuple[str | int] | Literal["*"] = (),
    use_shmem: bool = False,
):
    all_det_ids = [ch_id for ch_id in optmap_evt_columns if ch_id.isnumeric()]
    eff = np.array([get_channel_efficiency(int(ch_id), settings) for ch_id in all_det_ids])

    if chfilter != "*":
        chfilter = [str(ch) for ch in chfilter]  # normalize types
        optmap_det_ids = [det for det in all_det_ids if str(det) in chfilter]
    else:
        optmap_det_ids = all_det_ids

    log.info("creating empty optmaps")
    optmap_count = len(optmap_det_ids) + 1
    optmaps = [
        OpticalMap("all" if i == 0 else optmap_det_ids[i - 1], settings, use_shmem)
        for i in range(optmap_count)
    ]

    return all_det_ids, eff, optmaps, optmap_det_ids


@njit(cache=True)
def _compute_hit_maps(hitcounts, eff, rng, optmap_count, ch_idx_to_optmap):
    mask = np.zeros((hitcounts.shape[0], optmap_count), dtype=np.bool_)
    counts = hitcounts.sum(axis=1)
    for idx in range(hitcounts.shape[0]):
        if counts[idx] == 0:
            continue

        for ch_idx in range(hitcounts.shape[1]):
            c = rng.binomial(hitcounts[idx, ch_idx], eff[ch_idx])
            if c > 0:  # detected
                mask[idx, 0] = True
                mask_idx = ch_idx_to_optmap[ch_idx]
                if mask_idx > 0:
                    mask[idx, mask_idx] = True
    return mask


def _fill_hit_maps(
    optmaps: list[OpticalMap], loc, hitcounts: NDArray, eff: NDArray, ch_idx_to_map_idx, rng
):
    masks = _compute_hit_maps(hitcounts, eff, rng, len(optmaps), ch_idx_to_map_idx)

    for i in range(len(optmaps)):
        locm = loc[masks[:, i]]
        optmaps[i].fill_hits(locm)


def _count_multi_ph_detection(hitcounts) -> NDArray:
    hits_per_primary = hitcounts.sum(axis=1)
    bins = np.arange(0, hits_per_primary.max() + 1.5) - 0.5
    return np.histogram(hits_per_primary, bins)[0]


def _fit_multi_ph_detection(hits_per_primary) -> float:
    if len(hits_per_primary) <= 2:  # have only 0 and 1 hits, can't fit (and also don't need to).
        return np.inf

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


def _create_optical_maps_process_init(optmaps, log_level) -> None:
    # need to use shared global state. passing the shared memory arrays via "normal" arguments to
    # the worker function is not supported...
    global _shared_optmaps  # noqa: PLW0603
    _shared_optmaps = optmaps

    # setup logging in the worker process.
    setup_log(log_level, multiproc=True)


def _create_optical_maps_process(
    optmap_events_fn, buffer_len, all_det_ids, eff, ch_idx_to_map_idx
) -> None:
    log.info("started worker task for %s", optmap_events_fn)
    x = _create_optical_maps_chunk(
        optmap_events_fn,
        buffer_len,
        all_det_ids,
        eff,
        _shared_optmaps,
        ch_idx_to_map_idx,
    )
    log.info("finished worker task for %s", optmap_events_fn)
    return tuple(int(i) for i in x)


def _create_optical_maps_chunk(
    optmap_events_fn, buffer_len, all_det_ids, eff, optmaps, ch_idx_to_map_idx
) -> None:
    optmap_events_it = read_optmap_evt(optmap_events_fn, buffer_len)

    hits_per_primary = np.zeros(10, dtype=np.int64)
    hits_per_primary_len = 0
    rng = np.random.default_rng()
    for it_count, (events_lgdo, events_entry, event_n_rows) in enumerate(optmap_events_it):
        assert (it_count == 0) == (events_entry == 0)
        optmap_events = events_lgdo.view_as("pd").iloc[0:event_n_rows]
        hitcounts = optmap_events[all_det_ids].to_numpy()
        loc = optmap_events[["xloc", "yloc", "zloc"]].to_numpy()

        log.debug("filling vertex histogram (%d)", it_count)
        optmaps[0].fill_vertex(loc)

        log.debug("filling hits histogram (%d)", it_count)
        _fill_hit_maps(optmaps, loc, hitcounts, eff, ch_idx_to_map_idx, rng)
        hpp = _count_multi_ph_detection(hitcounts)
        hits_per_primary_len = max(hits_per_primary_len, len(hpp))
        hits_per_primary[0 : len(hpp)] += hpp

    return hits_per_primary[0:hits_per_primary_len]


def create_optical_maps(
    optmap_events_fn: list[str],
    settings,
    buffer_len: int = int(5e6),
    chfilter=(),
    output_lh5_fn=None,
    after_save: Callable[[int, str, OpticalMap]] | None = None,
    check_after_create: bool = False,
    n_procs: int | None = 1,
) -> None:
    """
    Parameters
    ----------
    optmap_events_fn
        list of filenames to lh5 files with a table ``/optmap_evt`` with columns ``{x,y,z}loc``
        and one column (with numeric header) for each SiPM channel.
    chfilter
        tuple of detector ids that will be included in the resulting optmap. Those have to match
        the column names in ``optmap_events_fn``.
    n_procs
        number of processors, ``1`` for sequential mode, or ``None`` to use all processors.
    """
    if len(optmap_events_fn) == 0:
        msg = "no input files specified"
        raise ValueError(msg)

    use_shmem = n_procs > 1

    optmap_evt_columns = list(
        lh5.read(EVT_TABLE_NAME, optmap_events_fn[0], start_row=0, n_rows=1).keys()
    )  # peek into the (first) file to find column names.
    all_det_ids, eff, optmaps, optmap_det_ids = _optmaps_for_channels(
        optmap_evt_columns, settings, chfilter=chfilter, use_shmem=use_shmem
    )

    # indices for later use in _compute_hit_maps.
    ch_idx_to_map_idx = np.array(
        [optmap_det_ids.index(d) + 1 if d in optmap_det_ids else -1 for d in all_det_ids]
    )
    assert np.sum(ch_idx_to_map_idx > 0) == len(optmaps) - 1

    log.info("creating optical map groups: %s", ", ".join(["all", *optmap_det_ids]))

    q = []

    # sequential mode.
    if not use_shmem:
        for fn in optmap_events_fn:
            q.append(
                _create_optical_maps_chunk(
                    fn, buffer_len, all_det_ids, eff, optmaps, ch_idx_to_map_idx
                )
            )
    else:
        ctx = mp.get_context("forkserver")
        for i in range(len(optmaps)):
            optmaps[i]._mp_preinit(ctx, vertex=(i == 0))

        # note: errors thrown in initializer will make the main process hang in an endless loop.
        # unfortunately, we cannot pass the objects later, as they contain shmem/array handles.
        pool = ctx.Pool(
            n_procs,
            initializer=_create_optical_maps_process_init,
            initargs=(optmaps, log.getEffectiveLevel()),
        )

        pool_results = []
        for fn in optmap_events_fn:
            r = pool.apply_async(
                _create_optical_maps_process,
                args=(fn, buffer_len, all_det_ids, eff, ch_idx_to_map_idx),
            )
            pool_results.append((r, fn))

        pool.close()
        for r, fn in pool_results:
            try:
                q.append(np.array(r.get()))
            except BaseException as e:
                msg = f"error while processing file {fn}"
                raise RuntimeError(msg) from e  # re-throw errors of workers.
        log.debug("got all worker results")
        pool.join()
        log.info("joined worker process pool")

    # merge hitcounts.
    if len(q) != len(optmap_events_fn):
        log.error("got %d results for %d files", len(q), len(optmap_events_fn))
    hits_per_primary = np.zeros(10, dtype=np.int64)
    hits_per_primary_len = 0
    for hitcounts in q:
        hits_per_primary[0 : len(hitcounts)] += hitcounts
        hits_per_primary_len = max(hits_per_primary_len, len(hitcounts))

    hits_per_primary = hits_per_primary[0:hits_per_primary_len]
    hits_per_primary_exponent = _fit_multi_ph_detection(hits_per_primary)

    # all maps share the same vertex histogram.
    for i in range(1, len(optmaps)):
        optmaps[i].h_vertex = optmaps[0].h_vertex

    log.info("computing probability and storing to %s", output_lh5_fn)
    for i in range(len(optmaps)):
        optmaps[i].create_probability()
        if check_after_create:
            optmaps[i].check_histograms()
        group = "all" if i == 0 else "_" + optmap_det_ids[i - 1]
        if output_lh5_fn is not None:
            optmaps[i].write_lh5(lh5_file=output_lh5_fn, group=group)

        if after_save is not None:
            after_save(i, group, optmaps[i])

        optmaps[i] = None  # clear some memory.

    if output_lh5_fn is not None:
        lh5.write(Array(hits_per_primary), "_hitcounts", lh5_file=output_lh5_fn)
        lh5.write(Scalar(hits_per_primary_exponent), "_hitcounts_exp", lh5_file=output_lh5_fn)


def list_optical_maps(lh5_file: str) -> list[str]:
    maps = lh5.ls(lh5_file)
    return [m for m in maps if m not in ("_hitcounts", "_hitcounts_exp")]


def merge_optical_maps(map_l5_files: list[str], output_lh5_fn: str, settings) -> None:
    store = LH5Store(keep_open=True)

    # verify that we have the same maps in all files.
    all_det_ntuples = None
    for optmap_fn in map_l5_files:
        det_ntuples = list_optical_maps(optmap_fn)
        if all_det_ntuples is not None and det_ntuples != all_det_ntuples:
            msg = "available optical maps in input files differ"
            raise ValueError(msg)
        all_det_ntuples = det_ntuples

    log.info("merging optical map groups: %s", ", ".join(all_det_ntuples))

    def _edges_eq(e1: tuple[NDArray], e2: tuple[NDArray]):
        return len(e1) == len(e2) and all(np.all(x1 == x2) for x1, x2 in zip(e1, e2))

    # merge maps one-by-one.
    for d in all_det_ntuples:
        merged_map = OpticalMap.create_empty(d, settings)
        merged_nr_gen = merged_map.h_vertex
        merged_nr_det = merged_map.h_hits

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
        merged_map.check_histograms(include_prefix=(len(all_det_ntuples) > 1))
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

    # re-calculate hitcounts exponent.
    hits_per_primary_exponent = _fit_multi_ph_detection(hits_per_primary)
    lh5.write(Scalar(hits_per_primary_exponent), "_hitcounts_exp", lh5_file=output_lh5_fn)


def check_optical_map(map_l5_file: str):
    for submap in list_optical_maps(map_l5_file):
        # TODO: check submaps consistency
        OpticalMap.load_from_file(map_l5_file, submap).check_histograms(include_prefix=True)