from __future__ import annotations

import logging

import awkward as ak
import numba
import numpy as np
from lgdo import Array

from reboost.hpge.drift_time import calculate_drift_times
from reboost.hpge.utils import ReadHPGeMap
from reboost.math.functions import piecewise_linear_activeness
from reboost.shape.cluster import apply_cluster, cluster_by_step_length

log = logging.getLogger(__name__)


def r90(edep: ak.Array, xloc: ak.Array, yloc: ak.Array, zloc: ak.Array) -> Array:
    """Computes R90 for each hit in a ged.

    Parameters
    ----------
    edep
        awkward array of energy
    xloc
        awkward array of x coordinate position
    yloc
        awkward array of y coordinate position
    zloc
        awkward array of z coordinate position

    Returns
    -------
    r90
    """
    tot_energy = ak.sum(edep, axis=-1, keepdims=True)

    def eweight_mean(field, energy):
        return ak.sum(energy * field, axis=-1, keepdims=True) / tot_energy

    # Compute distance of each edep to the weighted mean
    dist = np.sqrt(
        (xloc - eweight_mean(edep, xloc)) ** 2
        + (yloc - eweight_mean(edep, yloc)) ** 2
        + (zloc - eweight_mean(edep, zloc)) ** 2
    )

    # Sort distances and corresponding edep within each event
    sorted_indices = ak.argsort(dist, axis=-1)
    sorted_dist = dist[sorted_indices]
    sorted_edep = edep[sorted_indices]

    def cumsum(layout, **_kwargs):
        if layout.is_numpy:
            return ak.contents.NumpyArray(np.cumsum(layout.data))

        return None

    # Calculate the cumulative sum of energies for each event
    cumsum_edep = ak.transform(
        cumsum, sorted_edep
    )  # Implement cumulative sum over whole jagged array
    if len(edep) == 1:
        cumsum_edep_corrected = cumsum_edep
    else:
        cumsum_edep_corrected = (
            cumsum_edep[1:] - cumsum_edep[:-1, -1]
        )  # correct to get cumsum of each lower level array
        cumsum_edep_corrected = ak.concatenate(
            [
                cumsum_edep[:1],  # The first element of the original cumsum is correct
                cumsum_edep_corrected,
            ]
        )

    threshold = 0.9 * tot_energy
    r90_indices = ak.argmax(cumsum_edep_corrected >= threshold, axis=-1, keepdims=True)
    r90 = sorted_dist[r90_indices]

    return Array(ak.flatten(r90).to_numpy())


@numba.njit(cache=True)
def psa(t1: np.float64, e1: np.float64, t2: np.float64, e2: np.float64) -> np.float64:
    return abs(t1 - t2) / e_scaler(e1, e2)


@numba.njit(cache=True)
def e_scaler(e1: np.float64, e2: np.float64) -> np.float64:
    return 1 / np.sqrt(e1 * e2)


def do_cluster(grouped_data: ak.Array, cluster_size_mm: np.float64, drift_time_map: ReadHPGeMap):
    cluster_indices = cluster_by_step_length(
        grouped_data["evtid"],
        grouped_data["xloc"] * 1000,  # Convert to mm
        grouped_data["yloc"] * 1000,  # Convert to mm
        grouped_data["zloc"] * 1000,  # Convert to mm
        grouped_data["dist_to_surf"] * 1000,  # Convert to mm
        threshold=cluster_size_mm,
    )

    clustered_data = {
        f"{field}": apply_cluster(cluster_indices, grouped_data[f"{field}"]).view_as("ak")
        for field in ["edep", "xloc", "yloc", "zloc", "time", "dist_to_surf"]
    }
    clustered_data["activeness"] = piecewise_linear_activeness(
        clustered_data["dist_to_surf"], 0.5, 0.5
    ).view_as("ak")

    cluster_energy = ak.sum(clustered_data["edep"] * clustered_data["activeness"], axis=-1)
    cluster_energy_shaped = ak.broadcast_arrays(clustered_data["edep"], cluster_energy)[1]
    weights = ak.where(cluster_energy_shaped > 0, clustered_data["edep"] / cluster_energy_shaped, 0)

    clusters = {
        f"{field}": ak.sum(clustered_data[f"{field}"] * weights, axis=-1)
        for field in ["xloc", "yloc", "zloc", "time", "dist_to_surf", "activeness"]
    }
    clusters["edep"] = cluster_energy

    clusters["cluster_size"] = (
        ak.sum(
            weights
            * (
                (
                    clustered_data["xloc"]
                    - ak.broadcast_arrays(clustered_data["xloc"], clusters["xloc"])[1]
                )
                ** 2
                + (
                    clustered_data["yloc"]
                    - ak.broadcast_arrays(clustered_data["yloc"], clusters["yloc"])[1]
                )
                ** 2
                + (
                    clustered_data["zloc"]
                    - ak.broadcast_arrays(clustered_data["zloc"], clusters["zloc"])[1]
                )
                ** 2
            ),
            axis=-1,
        )
        ** 0.5
    )

    clusters["drift_time"] = calculate_drift_times(
        clusters["xloc"], clusters["yloc"], clusters["zloc"], drift_time_map
    )

    return ak.Array(clusters)


@numba.njit(cache=True)
def psa_heuristic_numba(
    drift_times: np.array, energies: np.array, event_offsets: np.array
) -> np.array:
    num_events = len(event_offsets) - 1
    dt_heuristic_output = np.zeros(num_events, dtype=np.float64)

    for evt_idx in range(num_events):
        start, end = event_offsets[evt_idx], event_offsets[evt_idx + 1]
        if start == end:
            continue

        event_energies = energies[start:end]
        event_drift_times = drift_times[start:end]

        valid_indices = np.where(event_energies > 0)[0]
        if len(valid_indices) < 2:
            continue

        filtered_drift_times = event_drift_times[valid_indices]
        filtered_energies = event_energies[valid_indices]
        nhits = len(event_drift_times)

        sorted_indices = np.argsort(filtered_drift_times)
        sorted_drift_times = filtered_drift_times[sorted_indices]
        sorted_energies = filtered_energies[sorted_indices]

        max_identify = 0
        for mkr in range(1, nhits):
            e1 = np.sum(sorted_energies[:mkr])
            e2 = np.sum(sorted_energies[mkr:])

            # when mkr == nhits, e1 = sum(sorted_energies) and e2 = 0
            if e1 > 0 and e2 > 0:
                t1 = np.sum(sorted_drift_times[:mkr] * sorted_energies[:mkr]) / e1
                t2 = np.sum(sorted_drift_times[mkr:] * sorted_energies[mkr:]) / e2

                identify = psa(t1, e1, t2, e2)
                max_identify = max(max_identify, identify)

        dt_heuristic_output[evt_idx] = max_identify

    return dt_heuristic_output


def dt_heuristic(data: ak.Array, drift_time_map: ReadHPGeMap) -> Array:
    # Test if the data has activeness
    if "activeness" not in data.fields:
        activeness = piecewise_linear_activeness(
            data["dist_to_surf"], fccd=0.5 / 1000, tl=0.5 / 1000
        ).view_as("ak")
        energies = data["edep"] * activeness
    else:
        energies = data["edep"]
    energies_flat = ak.flatten(energies).to_numpy()

    drift_times = calculate_drift_times(data.xloc, data.yloc, data.zloc, drift_time_map)
    drift_times_flat = ak.flatten(drift_times).to_numpy()

    event_offsets = np.append(0, np.cumsum(ak.num(drift_times)))

    dt_heuristic_output = psa_heuristic_numba(drift_times_flat, energies_flat, event_offsets)

    return Array(dt_heuristic_output)
