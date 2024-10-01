from __future__ import annotations

import logging

import legendoptics.scintillate as sc
import numpy as np
import pint
from legendoptics import lar
from lgdo import lh5
from lgdo.lh5 import LH5Iterator, LH5Store
from lgdo.types import Array, Table
from numba import njit, prange
from numpy.lib.recfunctions import structured_to_unstructured

from .numba_pdg import numba_pdgid_funcs

log = logging.getLogger(__name__)


OPTMAP_ANY_CH = -1
OPTMAP_SUM_CH = -2


def open_optmap(optmap_fn: str):
    store = LH5Store(keep_open=True)

    maps = lh5.ls(optmap_fn)
    det_ntuples = [m for m in maps if m not in ("all", "_hitcounts", "_hitcounts_exp")]
    detids = np.array([int(m.lstrip("_")) for m in det_ntuples])
    detidx = np.arange(0, detids.shape[0])

    optmap_all = store.read("/all/p_det", optmap_fn)[0]
    optmap_edges = tuple([b.edges for b in optmap_all.binning])

    optmap_weights = np.empty(
        (detidx.shape[0] + 2, *optmap_all.weights.nda.shape), dtype=np.float64
    )
    # normalize all negative components to 0.
    # TODO: somehow get statistics on missing voxels that would have been needed
    optmap_weights[OPTMAP_ANY_CH] = optmap_all.weights.nda
    optmap_weights[OPTMAP_ANY_CH][optmap_weights[OPTMAP_ANY_CH] < 0] = 0
    for i, nt in zip(detidx, det_ntuples):
        optmap = store.read(f"/{nt}/p_det", optmap_fn)[0]
        optmap_weights[i] = optmap.weights.nda
        optmap_weights[i][optmap_weights[i] < 0] = 0

    # if we have any individual channels registered, the sum is potentially larger than the
    # probability to find _any_ hit.
    if len(detidx) != 0:
        optmap_weights[OPTMAP_SUM_CH] = np.sum(optmap_weights[0:-2], axis=0)
        optmap_weights[OPTMAP_SUM_CH][optmap_weights[-2] < 0] = 0
    else:
        detidx = np.array([OPTMAP_ANY_CH])
        detids = np.array([0])
        optmap_weights[OPTMAP_SUM_CH] = optmap_weights[OPTMAP_ANY_CH]

    # give this check some numerical slack.
    if np.any(optmap_weights[-2] - optmap_weights[-1] < -1e-15):
        msg = "optical map does not fulfill relation sum(p_i) >= p_any"
        raise ValueError(msg)

    # get the exponent from the optical map file
    optmap_multi_det_exp = store.read("/_hitcounts_exp", optmap_fn)[0].value
    assert isinstance(optmap_multi_det_exp, float)

    return detids, detidx, optmap_edges, optmap_weights, optmap_multi_det_exp


def iterate_stepwise_depositions(
    edep_df: np.rec.recarray,
    optmap_for_convolve,
    scint_mat_params: sc.ComputedScintParams,
    rng: np.random.Generator = None,
):
    # those np functions are not supported by numba, but needed for efficient array access below.
    x0 = structured_to_unstructured(edep_df[["xloc_pre", "yloc_pre", "zloc_pre"]], np.float64)
    x1 = structured_to_unstructured(edep_df[["xloc_post", "yloc_post", "zloc_post"]], np.float64)

    rng = np.random.default_rng() if rng is None else rng

    output_map, res = _iterate_stepwise_depositions(
        edep_df, x0, x1, rng, *optmap_for_convolve, scint_mat_params
    )
    if res["oob"] > 0:
        log.warning(
            "had edep out of map bounds: %d (%.2f%%)", res["oob"], (res["oob"] / res["ib"]) * 100
        )
    log.debug(
        "VUV_primary %d ->hits_any %d ->hits %d (%.2f %% primaries detected)",
        res["vuv_primary"],
        res["hits_any"],
        res["hits"],
        (res["hits_any"] / res["vuv_primary"]) * 100,
    )
    log.debug("hits/hits_any %.2f", res["hits"] / res["hits_any"])
    return output_map


_pdg_func = numba_pdgid_funcs()


@njit
def _pdgid_to_particle(pdgid: int) -> sc.ParticleIndex:
    abs_pdgid = abs(pdgid)
    if abs_pdgid == 1000020040:
        return sc.PARTICLE_INDEX_ALPHA
    if abs_pdgid == 1000010020:
        return sc.PARTICLE_INDEX_DEUTERON
    if abs_pdgid == 1000010030:
        return sc.PARTICLE_INDEX_TRITON
    if _pdg_func.is_nucleus(pdgid):
        return sc.PARTICLE_INDEX_ION
    return sc.PARTICLE_INDEX_ELECTRON


# - run with NUMBA_FULL_TRACEBACKS=1 NUMBA_BOUNDSCHECK=1 for testing/checking
# - cache=True does not work with outer prange, i.e. loading the cached file fails (numba bug?)
# - the output dictionary is not threadsafe, so parallel=True is not working with it.
@njit(parallel=False, nogil=True, cache=True)
def _iterate_stepwise_depositions(
    edep_df,
    x0,
    x1,
    rng,
    detids,
    detidx,
    optmap_edges,
    optmap_weights,
    optmap_multi_det_exp,
    scint_mat_params: sc.ComputedScintParams,
):
    pdgid_map = {}
    output_map = {}
    oob = ib = ph_cnt = ph_det = ph_det2 = 0  # for statistics
    for rowid in prange(edep_df.shape[0]):
        # if rowid % 100000 == 0:
        #     print(rowid)
        t = edep_df[rowid]

        # get the particle information.
        if t.particle not in pdgid_map:
            pdgid_map[t.particle] = (_pdgid_to_particle(t.particle), _pdg_func.charge(t.particle))

        # do the scintillation.
        part, charge = pdgid_map[t.particle]
        scint_times = sc.scintillate(
            scint_mat_params,
            x0[rowid],
            x1[rowid],
            t.v_pre,
            t.v_post,
            t.time,
            part,
            charge,
            t.edep,
            rng,
        )
        if scint_times.shape[0] == 0:  # short-circuit if we have no photons at all.
            continue
        ph_cnt += scint_times.shape[0]

        # coordinates -> bins of the optical map.
        bins = np.empty((scint_times.shape[0], 3), dtype=np.int64)
        for j in range(3):
            bins[:, j] = np.digitize(scint_times[:, j + 1], optmap_edges[j])
            # normalize all out-of-bounds bins just to one end.
            bins[:, j][bins[:, j] == optmap_edges[j].shape[0]] = 0

        # there are _much_ less unique bins, unfortunately np.unique(..., axis=n) does not work
        # with numba; also np.sort(..., axis=n) also does not work.

        # get probabilities from map.
        hitcount = np.zeros((detidx.shape[0], bins.shape[0]), dtype=np.int64)
        for j in prange(bins.shape[0]):
            # note: subtract 1 from bins, to account for np.digitize output.
            cur_bins = (bins[j, 0] - 1, bins[j, 1] - 1, bins[j, 2] - 1)
            if cur_bins[0] == -1 or cur_bins[1] == -1 or cur_bins[2] == -1:
                oob += 1
                continue  # out-of-bounds of optmap
            ib += 1

            px_any = optmap_weights[OPTMAP_ANY_CH, cur_bins[0], cur_bins[1], cur_bins[2]]
            if px_any == 0.0:
                continue
            if rng.uniform() >= px_any:
                continue
            ph_det += 1
            # we detect this energy deposition; we should at least get one photon out here!

            detsel_size = rng.geometric(1 - np.exp(-optmap_multi_det_exp))

            px_sum = optmap_weights[OPTMAP_SUM_CH, cur_bins[0], cur_bins[1], cur_bins[2]]
            detp = np.empty(detidx.shape, dtype=np.float64)
            for d in detidx:
                # normalize so that sum(detp) = 1
                detp[d] = optmap_weights[d, cur_bins[0], cur_bins[1], cur_bins[2]] / px_sum

            # should be equivalent to rng.choice(detidx, size=(detsel_size, p=detp)
            detsel = detidx[
                np.searchsorted(np.cumsum(detp), rng.random(size=(detsel_size,)), side="right")
            ]
            for d in detsel:
                hitcount[d, j] += 1
            ph_det2 += detsel.shape[0]

        out_hits_len = np.sum(hitcount)
        if out_hits_len > 0:
            out_times = np.empty(out_hits_len, dtype=np.float64)
            out_det = np.empty(out_hits_len, dtype=np.int64)
            out_idx = 0
            for d in detidx:
                tx_max_plane = np.max(hitcount[d, :])
                # untangle the hitcount array in "planes" that only contain the given number of hits per channel.
                for hc_d_plane_cnt in range(1, tx_max_plane + 1):
                    hc_d_plane = hitcount[d, :] == hc_d_plane_cnt
                    hc_d_plane_len = np.sum(hc_d_plane)
                    if hc_d_plane_len == 0:
                        continue

                    out_times[out_idx : out_idx + hc_d_plane_len] = scint_times[hc_d_plane, 0]
                    out_det[out_idx : out_idx + hc_d_plane_len] = detids[d]
                    out_idx += hc_d_plane_len
            output_map[np.int64(rowid)] = (t.evtid, out_det, out_times)

    stats = {"oob": oob, "ib": ib, "vuv_primary": ph_cnt, "hits_any": ph_det, "hits": ph_det2}
    return output_map, stats


def get_output_table(output_map):
    ph_count_o = 0
    for _rawid, (_evtid, det, _times) in output_map.items():
        ph_count_o += det.shape[0]
    out_idx = 0
    out_evtid = np.empty(ph_count_o, dtype=np.int64)
    out_det = np.empty(ph_count_o, dtype=np.int64)
    out_times = np.empty(ph_count_o, dtype=np.float64)
    for _rawid, (evtid, det, times) in output_map.items():
        o_len = det.shape[0]
        out_evtid[out_idx : out_idx + o_len] = evtid
        out_det[out_idx : out_idx + o_len] = det
        out_times[out_idx : out_idx + o_len] = times
        out_idx += o_len

    tbl = Table({"evtid": Array(out_evtid), "detid": Array(out_det), "time": Array(out_times)})
    return ph_count_o, tbl


def convolve(
    map_file: str,
    edep_file: str,
    edep_path: str,
    material: str,
    output_file: str | None = None,
    buffer_len: int = int(1e6),
):
    if material not in ["lar", "pen"]:
        msg = f"unknown material {material} for scintillation"
        raise ValueError(msg)

    if material == "lar":
        scint_mat_params = sc.precompute_scintillation_params(
            lar.lar_scintillation_params(),
            lar.lar_lifetimes().as_tuple(),
        )
    elif material == "pen":
        scint_mat_params = sc.precompute_scintillation_params(
            lar.pen_scintillation_params(),
            (1 * pint.get_application_registry().ns),  # dummy!
        )

    log.info("opening map %s", map_file)
    optmap_for_convolve = open_optmap(map_file)

    log.info("opening energy deposition hit output %s", edep_file)
    it = LH5Iterator(edep_file, edep_path, buffer_len=buffer_len)
    for it_count, (edep_lgdo, edep_events, edep_n_rows) in enumerate(it):
        assert (it_count == 0) == (edep_events == 0)
        edep_df = edep_lgdo.view_as("pd").iloc[0:edep_n_rows].to_records()

        log.info("start event processing (%d)", it_count)
        output_map = iterate_stepwise_depositions(edep_df, optmap_for_convolve, scint_mat_params)

        log.info("store output photon hits (%d)", it_count)
        ph_count_o, tbl = get_output_table(output_map)
        log.debug(
            "output photons: %d energy depositions -> %d photons", len(output_map), ph_count_o
        )
        if output_file is not None:
            lh5.write(tbl, "optical", lh5_file=output_file, group="hit", wo_mode="append")
