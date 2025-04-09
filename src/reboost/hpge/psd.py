from __future__ import annotations

import logging

import awkward as ak
import numba
import numpy as np
from lgdo import Array

from reboost.hpge.drift_time import calculate_drift_times
from reboost.math.functions import piecewise_linear_activeness

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
def identification_metric(
    t1: np.float64, e1: np.float64, t2: np.float64, e2: np.float64
) -> np.float64:
    return abs(t1 - t2) / e_scaler(e1, e2)


@numba.njit(cache=True)
def e_scaler(e1: np.float64, e2: np.float64) -> np.float64:
    return 1 / np.sqrt(e1 * e2)


@numba.njit(cache=True)
def calculate_dt_heuristic(
    drift_times: np.array, energies: np.array, event_offsets: np.array
) -> np.array:
    r"""Computes the drift time heuristic pulse shape analysis (PSA) metric for each event based on drift time and energy of hits (steps or clusters) within a Ge detector.

    This function iterates over a set of events, extracts drift times and energies
    for each event, and identifies the drift time separation that maximizes
    the heuristic identification metric.

    Parameters
    ----------
    drift_times : np.array
        flattened array of drift times corresponding to hits in the detector.
    energies : np.array
        flattened array of energy depositions associated with each hit.
    event_offsets : np.array
        flattened array indicating the start and end indices of each event in `drift_times` and `energies`.

    Returns
    -------
    np.array
        Array containing the maximum PSA identification metric for each event.

    Notes
    -----
    - For each event, the drift times and corresponding energies are sorted in ascending order.
    - The function finds the optimal split point `m` that maximizes the **identification metric**:

      .. math::

          I = \\frac{|T_1 - T_2|}{E_{\text{scale}}(E_1, E_2)}

      where:

      - :math:`T_1 = \\frac{\sum_{i < m} t_i E_i}{\sum_{i < m} E_i}`  and
        :math:`T_2 = \\frac{\sum_{i \geq m} t_i E_i}{\sum_{i \geq m} E_i}`
        are the energy-weighted mean drift times of the two groups.
      - :math:`E_{\text{scale}}(E_1, E_2) = \\frac{1}{\sqrt{E_1 E_2}}`
        is the scaling factor.
    - The function iterates over all possible values of `m` and selects the maximum `I`.
    """
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

                identify = identification_metric(t1, e1, t2, e2)
                max_identify = max(max_identify, identify)

        dt_heuristic_output[evt_idx] = max_identify

    return dt_heuristic_output


def dt_heuristic(data: ak.Array, drift_time_map: dict) -> Array:
    """Computes the drift time heuristic pulse shape analysis (PSA) metric for each hit.

    This function calculates drift times for each hit (step or cluster)
    in a germanium detector and computes the heuristic metric based on drift time separation
    and energy weighting.

    Parameters
    ----------
    data : ak.Array
        An awkward array containing event step or cluster, including hit positions
        (`xloc`, `yloc`, `zloc`) and deposited energy (`edep`). Optionally, it may contain
        `activeness` values.
    drift_time_map : ReadHPGeMap
        Drift time HPGe map used to determine drift times for given spatial coordinates.

    Returns
    -------
    Array
        A LDGO Array containing the computed dt heuristic value for each event.

    Notes
    -----
    - If `activeness` is not present in `data`, it is computed using a piecewise linear function
      based on distance to the detector surface.
    - Drift times are computed using `calculate_drift_times()`, mapped from spatial coordinates.
    - The function flattens energy and drift time arrays to prepare for numba processing.
    - Then`calculate_dt_heuristic()` is called which does the computation.
    """
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

    dt_heuristic_output = calculate_dt_heuristic(drift_times_flat, energies_flat, event_offsets)

    return Array(dt_heuristic_output)
