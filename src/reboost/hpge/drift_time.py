from __future__ import annotations

import logging

import awkward as ak
import numpy as np

from reboost.hpge.utils import interpolate2D

log = logging.getLogger(__name__)


def calculate_drift_times(
    xdata: ak.Array, ydata: ak.Array, zdata: ak.Array, drift_time_map: dict
) -> ak.Array:
    """Calculate drift times for each hit (step/cluster) based on the provided drift time map.

    Args:
        xdata (ak.Array): X coordinates of the hits (step/cluster).
        ydata (ak.Array): Y coordinates of the hits (step/cluster).
        zdata (ak.Array): Z coordinates of the hits (step/cluster).
        drift_time_map (dict): Drift time map data.

    Returns
    -------
        ak.Array: Drift times for each cluster element.
    """
    # Flatten the input arrays to 1D and calculate relative positions
    position_flat_relative = [
        ak.flatten(axis_data).to_numpy() - drift_time_map["detector_position"][key]
        for axis_data, key in zip([xdata, ydata, zdata], ["x", "y", "z"])
    ]
    rdata = np.sqrt(position_flat_relative[0] ** 2 + position_flat_relative[1] ** 2)

    # Vectorized calculation of drift times for each cluster element
    drift_times_flat = np.vectorize(
        lambda dt_map, x, y, x_key, y_key, val_key: interpolate2D(
            dt_map, x, y, x_key, y_key, val_key
        )
    )(drift_time_map, rdata, position_flat_relative[2], "r", "z", "dt")

    return ak.unflatten(drift_times_flat, ak.num(xdata))
