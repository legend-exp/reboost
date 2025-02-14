from __future__ import annotations

import awkward as ak
import numpy as np

from reboost.hpge.utils import ReadDTFile


def calculate_drift_times(
    xdata: ak.Array, ydata: ak.Array, zdata: ak.Array, dt_file: ReadDTFile
) -> ak.Array:
    x_flat = ak.flatten(xdata).to_numpy() * 1000  # Convert to mm
    y_flat = ak.flatten(ydata).to_numpy() * 1000  # Convert to mm
    z_flat = ak.flatten(zdata).to_numpy() * 1000  # Convert to mm

    # Vectorized calculation of drift times for each cluster element
    drift_times_flat = np.vectorize(lambda x, y, z: dt_file.GetTime(x, y, z))(
        x_flat, y_flat, z_flat
    )

    return ak.unflatten(drift_times_flat, ak.num(xdata))
