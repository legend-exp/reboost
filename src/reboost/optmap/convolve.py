from __future__ import annotations

import logging
from typing import Literal, NamedTuple, TypeAlias

import awkward as ak
import lh5
import numpy as np
import pint
import pygeomoptics.scintillate as sc
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
    edges: tuple
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


def _warn_deposition_stats(res: dict) -> None:
    looped_steps = res["ib"] + res["oob"]
    if res["det_no_stats"] > 0:
        log.warning(
            "steps in optmap voxels without stats: %d (%.2f%%)",
            res["det_no_stats"],
            (res["det_no_stats"] / looped_steps) * 100 if looped_steps > 0 else 0.0,
        )
    if res["oob"] > 0:
        log.warning(
            "steps outside optmap domain: %d (%.2f%%)",
            res["oob"],
            (res["oob"] / looped_steps) * 100 if looped_steps > 0 else 0.0,
        )

    if res["vuv_primary_oob"] > 0:
        log.warning(
            "VUV_primary in voxels outside optmap domain: %d (%.2f%%)",
            res["vuv_primary_oob"],
            (res["vuv_primary_oob"] / res["vuv_primary_looped"]) * 100
            if res["vuv_primary_looped"] > 0
            else 0.0,
        )
    if res["vuv_primary_no_stats"] > 0:
        log.warning(
            "VUV_primary in voxels without optmap stats: %d (%.2f%%)",
            res["vuv_primary_no_stats"],
            (res["vuv_primary_no_stats"] / res["vuv_primary_looped"]) * 100
            if res["vuv_primary_looped"] > 0
            else 0.0,
        )


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
    max_pes_per_hit: int = -1,
    rng: np.random.Generator | None = None,
    return_pes_expectation_value: bool = False,
) -> ak.Array | tuple[ak.Array, NDArray]:
    if edep_hits.xloc.ndim == 1:
        msg = "the pe processors only support already reshaped output"
        raise ValueError(msg)

    rng = np.random.default_rng() if rng is None else rng
    counts = ak.num(edep_hits.num_scint_ph)
    output_array, exp_pes_array, max_ph_reached, res = _iterate_stepwise_depositions_numdet(
        edep_hits,
        rng,
        np.where(optmap.dets == det)[0][0],
        map_scaling,
        map_scaling_sigma,
        optmap.edges,
        optmap.weights,
        ak.sum(counts),
        max_pes_per_hit,
        return_pes_expectation_value,
    )

    _warn_deposition_stats(res)

    out = ak.unflatten(output_array, counts)
    if return_pes_expectation_value:
        return (out, max_ph_reached, exp_pes_array) if max_pes_per_hit > 0 else (out, exp_pes_array)
    if max_pes_per_hit > 0:
        return out, max_ph_reached
    return out


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
    max_pes_per_hit: int = -1,
    return_pes_expectation_value: bool = False,
):
    oob = ib = det_no_stats = 0
    vuv_primary_oob = vuv_primary_no_stats = vuv_primary_inb = 0
    output = np.empty(shape=output_length, dtype=np.int64)
    # p.e. expectation per row, at unit efficiency and before truncation
    expected_pes = np.empty(
        shape=len(edep_hits) if return_pes_expectation_value else 0, dtype=np.float64
    )
    has_max_ph_hit = np.full(shape=len(edep_hits), fill_value=False, dtype=np.bool)

    output_index = 0
    for rowid in range(len(edep_hits)):  # iterate hits
        hit = edep_hits[rowid]

        map_scaling_evt = map_scaling
        if map_scaling_sigma > 0:
            map_scaling_evt = rng.normal(loc=map_scaling, scale=map_scaling_sigma)

        # iterate steps inside the hit
        photons_in_hit = 0
        expected_pes_row = 0.0
        for si in range(len(hit.xloc)):
            capped = max_pes_per_hit > 0 and photons_in_hit >= max_pes_per_hit
            if capped and not return_pes_expectation_value:
                output[output_index] = 0
                output_index += 1
                has_max_ph_hit[rowid] = True
                continue

            vuv_step = hit.num_scint_ph[si]

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
                mapw = 0.0
                detp = 0.0  # out-of-bounds of optmap
                oob += 1
                vuv_primary_oob += vuv_step
            else:
                # get probabilities from map.
                mapw = optmap_weights[detidx, bins[0], bins[1], bins[2]]
                detp = mapw * map_scaling_evt
                if detp < 0:
                    det_no_stats += 1
                    vuv_primary_no_stats += vuv_step
                else:
                    vuv_primary_inb += vuv_step
                ib += 1

            if return_pes_expectation_value:
                expected_pes_row += 0.0 if mapw <= 0.0 else vuv_step * mapw
                if capped:
                    has_max_ph_hit[rowid] = True
                    output[output_index] = 0
                    output_index += 1
                    continue

            pois_cnt = 0 if detp <= 0.0 else rng.poisson(lam=vuv_step * detp)
            photons_in_hit += pois_cnt
            if max_pes_per_hit > 0 and photons_in_hit >= max_pes_per_hit:
                pois_cnt -= photons_in_hit - max_pes_per_hit
                has_max_ph_hit[rowid] = True
            output[output_index] = pois_cnt
            output_index += 1

        if return_pes_expectation_value:
            expected_pes[rowid] = expected_pes_row

    assert output_index == output_length

    return (
        output,
        expected_pes,
        has_max_ph_hit,
        {
            "oob": oob,
            "ib": ib,
            "det_no_stats": det_no_stats,
            "vuv_primary_looped": vuv_primary_inb + vuv_primary_oob + vuv_primary_no_stats,
            "vuv_primary_oob": vuv_primary_oob,
            "vuv_primary_no_stats": vuv_primary_no_stats,
        },
    )


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


ScintMaterial: TypeAlias = (
    Literal["lar", "pen", "fiber"] | tuple[sc.ScintConfig, tuple[pint.Quantity, ...]]
)


def _get_scint_params(material: ScintMaterial):
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
        raise ValueError(msg)  # noqa: TRY004 (this is not a typy error)
    return sc.precompute_scintillation_params(*material)
