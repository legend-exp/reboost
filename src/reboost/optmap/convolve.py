from __future__ import annotations

import logging
from typing import NamedTuple

import awkward as ak
import numba
import numpy as np
import pygeomoptics.scintillate as sc
from lgdo import lh5
from lgdo.types import Histogram
from numba import njit
from numpy.typing import NDArray
from pygeomoptics import fibers, lar, pen

from .numba_pdg import numba_pdgid_funcs

log = logging.getLogger(__name__)


OPTMAP_ANY_CH = -1


class OptmapForConvolve(NamedTuple):
    """A loaded optmap for convolving."""

    dets: NDArray
    detidx: NDArray
    edges: NDArray
    weights: NDArray


def open_optmap(optmap_fn: str) -> OptmapForConvolve:
    dets = lh5.ls(optmap_fn, "/channels/")
    detidx = np.arange(0, len(dets))

    optmap_all = lh5.read("/all/prob", optmap_fn)
    assert isinstance(optmap_all, Histogram)
    optmap_edges = tuple([b.edges for b in optmap_all.binning])

    ow = np.empty((detidx.shape[0] + 2, *optmap_all.weights.nda.shape), dtype=np.float64)
    # 0, ..., len(detidx)-1 AND OPTMAP_ANY_CH might be negative.
    ow[OPTMAP_ANY_CH] = optmap_all.weights.nda
    for i, nt in zip(detidx, dets, strict=True):
        optmap = lh5.read(f"/{nt}/prob", optmap_fn)
        assert isinstance(optmap, Histogram)
        ow[i] = optmap.weights.nda

    # if we have any individual channels registered, the sum is potentially larger than the
    # probability to find _any_ hit.
    if len(detidx) != 0:
        map_sum = np.sum(ow[0:-2], axis=0, where=(ow[0:-2] >= 0))
        assert not np.any(map_sum < 0)

        # give this check some numerical slack.
        if np.any(
            np.abs(map_sum[ow[OPTMAP_ANY_CH] >= 0] - ow[OPTMAP_ANY_CH][ow[OPTMAP_ANY_CH] >= 0])
            < -1e-15
        ):
            msg = "optical map does not fulfill relation sum(p_i) >= p_any"
            raise ValueError(msg)
    else:
        detidx = np.array([OPTMAP_ANY_CH])
        dets = ["all"]

    # check the exponent from the optical map file
    if "_hitcounts_exp" in lh5.ls(optmap_fn):
        msg = "found _hitcounts_exp which is not supported any more"
        raise RuntimeError(msg)

    dets = np.array([d.replace("/channels/", "") for d in dets])

    return OptmapForConvolve(dets, detidx, optmap_edges, ow)


def open_optmap_single(optmap_fn: str, spm_det: str) -> OptmapForConvolve:
    # check the exponent from the optical map file
    if "_hitcounts_exp" in lh5.ls(optmap_fn):
        msg = "found _hitcounts_exp which is not supported any more"
        raise RuntimeError(msg)

    h5_path = f"channels/{spm_det}" if spm_det != "all" else spm_det
    optmap = lh5.read(f"/{h5_path}/prob", optmap_fn)
    assert isinstance(optmap, Histogram)
    ow = np.empty((1, *optmap.weights.nda.shape), dtype=np.float64)
    ow[0] = optmap.weights.nda
    optmap_edges = tuple([b.edges for b in optmap.binning])

    return OptmapForConvolve(np.array([spm_det]), np.array([0]), optmap_edges, ow)


def iterate_stepwise_depositions_pois(
    edep_hits: ak.Array,
    optmap: OptmapForConvolve,
    scint_mat_params: sc.ComputedScintParams,
    det: str,
    map_scaling: float = 1,
    map_scaling_sigma: float = 0,
    rng: np.random.Generator | None = None,
):
    if edep_hits.particle.ndim == 1:
        msg = "the pe processors only support already reshaped output"
        raise ValueError(msg)

    if det not in optmap.dets:
        msg = f"channel {det} not available in optical map (contains {optmap.dets})"
        raise ValueError(msg)

    rng = np.random.default_rng() if rng is None else rng
    res, output_list = _iterate_stepwise_depositions_pois(
        edep_hits,
        rng,
        np.where(optmap.dets == det)[0][0],
        map_scaling,
        map_scaling_sigma,
        optmap.edges,
        optmap.weights,
        scint_mat_params,
    )

    # convert the numba result back into an awkward array.
    builder = ak.ArrayBuilder()
    for r in output_list:
        with builder.list():
            for a in r:
                builder.extend(a)

    if res["det_no_stats"] > 0:
        log.warning(
            "had edep out in voxels without stats: %d",
            res["det_no_stats"],
        )
    if res["oob"] > 0:
        log.warning(
            "had edep out of map bounds: %d (%.2f%%)",
            res["oob"],
            (res["oob"] / (res["ib"] + res["oob"])) * 100,
        )
    log.debug(
        "VUV_primary %d ->hits %d (%.2f %% primaries detected in this channel)",
        res["vuv_primary"],
        res["hits"],
        (res["hits"] / res["vuv_primary"]) * 100,
    )
    return builder.snapshot()


def iterate_stepwise_depositions_scintillate(
    edep_hits: ak.Array,
    scint_mat_params: sc.ComputedScintParams,
    rng: np.random.Generator | None = None,
    mode: str = "no-fano",
):
    if edep_hits.particle.ndim == 1:
        msg = "the pe processors only support already reshaped output"
        raise ValueError(msg)

    rng = np.random.default_rng() if rng is None else rng
    counts = ak.num(edep_hits.edep)
    output_array = _iterate_stepwise_depositions_scintillate(
        edep_hits, rng, scint_mat_params, mode, ak.sum(counts)
    )

    return ak.unflatten(output_array, counts)


def iterate_stepwise_depositions_numdet(
    edep_hits: ak.Array,
    optmap: OptmapForConvolve,
    det: str,
    map_scaling: float = 1,
    map_scaling_sigma: float = 0,
    rng: np.random.Generator | None = None,
):
    if edep_hits.xloc.ndim == 1:
        msg = "the pe processors only support already reshaped output"
        raise ValueError(msg)

    rng = np.random.default_rng() if rng is None else rng
    counts = ak.num(edep_hits.num_scint_ph)
    output_array, res = _iterate_stepwise_depositions_numdet(
        edep_hits,
        rng,
        np.where(optmap.dets == det)[0][0],
        map_scaling,
        map_scaling_sigma,
        optmap.edges,
        optmap.weights,
        ak.sum(counts),
    )

    if res["det_no_stats"] > 0:
        log.warning(
            "had edep out in voxels without stats: %d",
            res["det_no_stats"],
        )
    if res["oob"] > 0:
        log.warning(
            "had edep out of map bounds: %d (%.2f%%)",
            res["oob"],
            (res["oob"] / (res["ib"] + res["oob"])) * 100,
        )

    return ak.unflatten(output_array, counts)


def iterate_stepwise_depositions_times(
    edep_hits: ak.Array,
    scint_mat_params: sc.ComputedScintParams,
    rng: np.random.Generator | None = None,
):
    if edep_hits.particle.ndim == 1:
        msg = "the pe processors only support already reshaped output"
        raise ValueError(msg)

    rng = np.random.default_rng() if rng is None else rng
    counts = ak.sum(edep_hits.num_det_ph, axis=1)
    output_array = _iterate_stepwise_depositions_times(
        edep_hits, rng, scint_mat_params, ak.sum(counts)
    )

    return ak.unflatten(output_array, counts)


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


__counts_per_bin_key_type = numba.types.UniTuple(numba.types.int64, 3)


# - run with NUMBA_FULL_TRACEBACKS=1 NUMBA_BOUNDSCHECK=1 for testing/checking
# - cache=True does not work with outer prange, i.e. loading the cached file fails (numba bug?)
# - the output dictionary is not threadsafe, so parallel=True is not working with it.
@njit(parallel=False, nogil=True, cache=True)
def _iterate_stepwise_depositions_pois(
    edep_hits,
    rng,
    detidx: int,
    map_scaling: float,
    map_scaling_sigma: float,
    optmap_edges,
    optmap_weights,
    scint_mat_params: sc.ComputedScintParams,
):
    pdgid_map = {}
    oob = ib = ph_cnt = ph_det2 = det_no_stats = 0  # for statistics
    output_list = []

    for rowid in range(len(edep_hits)):  # iterate hits
        hit = edep_hits[rowid]
        hit_output = []

        map_scaling_evt = map_scaling
        if map_scaling_sigma > 0:
            map_scaling_evt = rng.normal(loc=map_scaling, scale=map_scaling_sigma)

        assert len(hit.particle) == len(hit.num_scint_ph)
        # iterate steps inside the hit
        for si in range(len(hit.particle)):
            loc = np.array([hit.xloc[si], hit.yloc[si], hit.zloc[si]])
            # coordinates -> bins of the optical map.
            bins = np.empty(3, dtype=np.int64)
            for j in range(3):
                bins[j] = np.digitize(loc[j], optmap_edges[j])
                # normalize all out-of-bounds bins just to one end.
                if bins[j] == optmap_edges[j].shape[0]:
                    bins[j] = 0

            # note: subtract 1 from bins, to account for np.digitize output.
            cur_bins = (bins[0] - 1, bins[1] - 1, bins[2] - 1)
            if cur_bins[0] == -1 or cur_bins[1] == -1 or cur_bins[2] == -1:
                oob += 1
                continue  # out-of-bounds of optmap
            ib += 1

            # get probabilities from map.
            detp = optmap_weights[detidx, cur_bins[0], cur_bins[1], cur_bins[2]] * map_scaling_evt
            if detp < 0.0:
                det_no_stats += 1
                continue

            pois_cnt = rng.poisson(lam=hit.num_scint_ph[si] * detp)
            ph_cnt += hit.num_scint_ph[si]
            ph_det2 += pois_cnt

            # get the particle information.
            particle = hit.particle[si]
            if particle not in pdgid_map:
                pdgid_map[particle] = (_pdgid_to_particle(particle), _pdg_func.charge(particle))
            part, _charge = pdgid_map[particle]

            # get time spectrum.
            # note: we assume "immediate" propagation after scintillation.
            scint_times = sc.scintillate_times(scint_mat_params, part, pois_cnt, rng) + hit.time[si]

            hit_output.append(scint_times)

        output_list.append(hit_output)

    stats = {
        "oob": oob,
        "ib": ib,
        "vuv_primary": ph_cnt,
        "hits": ph_det2,
        "det_no_stats": det_no_stats,
    }
    return stats, output_list


# - run with NUMBA_FULL_TRACEBACKS=1 NUMBA_BOUNDSCHECK=1 for testing/checking
# - cache=True does not work with outer prange, i.e. loading the cached file fails (numba bug?)
@njit(parallel=False, nogil=True, cache=True)
def _iterate_stepwise_depositions_scintillate(
    edep_hits, rng, scint_mat_params: sc.ComputedScintParams, mode: str, output_length: int
):
    pdgid_map = {}
    output = np.empty(shape=output_length, dtype=np.int64)

    output_index = 0
    for rowid in range(len(edep_hits)):  # iterate hits
        hit = edep_hits[rowid]
        for si in range(len(hit.particle)):  # iterate steps inside the hit
            # get the particle information.
            particle = hit.particle[si]
            if particle not in pdgid_map:
                pdgid_map[particle] = (_pdgid_to_particle(particle), _pdg_func.charge(particle))
            part, _charge = pdgid_map[particle]

            # do the scintillation.
            num_phot = sc.scintillate_numphot(
                scint_mat_params,
                part,
                hit.edep[si],
                rng,
                emission_term_model=("poisson" if mode == "no-fano" else "normal_fano"),
            )
            output[output_index] = num_phot
            output_index += 1

    assert output_index == output_length
    return output


# - run with NUMBA_FULL_TRACEBACKS=1 NUMBA_BOUNDSCHECK=1 for testing/checking
# - cache=True does not work with outer prange, i.e. loading the cached file fails (numba bug?)
@njit(parallel=False, nogil=True, cache=True)
def _iterate_stepwise_depositions_numdet(
    edep_hits,
    rng,
    detidx: int,
    map_scaling: float,
    map_scaling_sigma: float,
    optmap_edges,
    optmap_weights,
    output_length: int,
):
    oob = ib = det_no_stats = 0
    output = np.empty(shape=output_length, dtype=np.int64)

    output_index = 0
    for rowid in range(len(edep_hits)):  # iterate hits
        hit = edep_hits[rowid]

        map_scaling_evt = map_scaling
        if map_scaling_sigma > 0:
            map_scaling_evt = rng.normal(loc=map_scaling, scale=map_scaling_sigma)

        # iterate steps inside the hit
        for si in range(len(hit.xloc)):
            loc = np.array([hit.xloc[si], hit.yloc[si], hit.zloc[si]], dtype=np.float64)
            # coordinates -> bins of the optical map.
            bins = np.empty(3, dtype=np.int64)
            for j in range(3):
                edges = optmap_edges[j].astype(np.float64)
                start = edges[0]
                width = edges[1] - edges[0]
                nbins = edges.shape[0] - 1
                bins[j] = int((loc[j] - start) / width)

                if bins[j] < 0 or bins[j] >= nbins:
                    bins[j] = -1  # normalize all out-of-bounds bins just to one end.

            if bins[0] == -1 or bins[1] == -1 or bins[2] == -1:
                detp = 0.0  # out-of-bounds of optmap
                oob += 1
            else:
                # get probabilities from map.
                detp = optmap_weights[detidx, bins[0], bins[1], bins[2]] * map_scaling_evt
                if detp < 0:
                    det_no_stats += 1
                ib += 1

            pois_cnt = 0 if detp <= 0.0 else rng.poisson(lam=hit.num_scint_ph[si] * detp)
            output[output_index] = pois_cnt
            output_index += 1

    assert output_index == output_length
    return output, {"oob": oob, "ib": ib, "det_no_stats": det_no_stats}


# - run with NUMBA_FULL_TRACEBACKS=1 NUMBA_BOUNDSCHECK=1 for testing/checking
# - cache=True does not work with outer prange, i.e. loading the cached file fails (numba bug?)
# - the output dictionary is not threadsafe, so parallel=True is not working with it.
@njit(parallel=False, nogil=True, cache=True)
def _iterate_stepwise_depositions_times(
    edep_hits, rng, scint_mat_params: sc.ComputedScintParams, output_length: int
):
    pdgid_map = {}
    output = np.empty(shape=output_length, dtype=np.float64)

    output_index = 0
    for rowid in range(len(edep_hits)):  # iterate hits
        hit = edep_hits[rowid]

        assert len(hit.particle) == len(hit.num_det_ph)
        # iterate steps inside the hit
        for si in range(len(hit.particle)):
            pois_cnt = hit.num_det_ph[si]
            if pois_cnt <= 0:
                continue

            # get the particle information.
            particle = hit.particle[si]
            if particle not in pdgid_map:
                pdgid_map[particle] = (_pdgid_to_particle(particle), _pdg_func.charge(particle))
            part, _charge = pdgid_map[particle]

            # get time spectrum.
            # note: we assume "immediate" propagation after scintillation.
            scint_times = sc.scintillate_times(scint_mat_params, part, pois_cnt, rng) + hit.time[si]
            assert len(scint_times) == pois_cnt
            output[output_index : output_index + len(scint_times)] = scint_times
            output_index += len(scint_times)

    assert output_index == output_length
    return output


def _get_scint_params(material: str):
    if material == "lar":
        return sc.precompute_scintillation_params(
            lar.lar_scintillation_params(),
            lar.lar_lifetimes().as_tuple(),
        )
    if material == "pen":
        return sc.precompute_scintillation_params(
            pen.pen_scintillation_params(),
            (pen.pen_scint_timeconstant(),),
        )
    if material == "fiber":
        return sc.precompute_scintillation_params(
            fibers.fiber_core_scintillation_params(),
            (fibers.fiber_wls_timeconstant(),),
        )
    if isinstance(material, str):
        msg = f"unknown material {material} for scintillation"
        raise ValueError(msg)
    return sc.precompute_scintillation_params(*material)
