from __future__ import annotations

import logging

import awkward as ak
import numpy as np

from reboost.hpge.utils import ReadHPGeMap

log = logging.getLogger(__name__)


def calculate_drift_times(
    xdata: ak.Array, ydata: ak.Array, zdata: ak.Array, drift_time_map: ReadHPGeMap
) -> ak.Array:
    position_flat = [ak.flatten(axis_data).to_numpy() for axis_data in [xdata, ydata, zdata]]

    # Vectorized calculation of drift times for each cluster element
    drift_times_flat = np.vectorize(lambda x, y, z: drift_time_map.get_map_value(x, y, z))(
        *position_flat
    )

    return ak.unflatten(drift_times_flat, ak.num(xdata))
