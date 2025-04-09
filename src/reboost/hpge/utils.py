from __future__ import annotations

from math import floor

import numpy as np
from lgdo.lh5 import LH5Store


def interpolate2D(
    map_dict: dict,
    x: np.float64,
    y: np.float64,
    x_key: str = "x",
    y_key: str = "y",
    val_key: str = "val",
    is_relative: bool = True,
) -> np.float64:
    """Interpolates the drift time map for a given x and y position.

    Args:
        map_dict (dict): The drift time map data.
        x (np.float64): The x position.
        y (np.float64): The y position.

    Returns
    -------
        np.float64: The interpolated map value.
    """
    if not is_relative:
        x -= map_dict["detector_position"][x_key]
        y -= map_dict["detector_position"][y_key]

    dx = map_dict[f"d{x_key}"]
    dy = map_dict[f"d{y_key}"]
    min_x = map_dict[f"min_{x_key}"]
    min_y = map_dict[f"min_{y_key}"]
    max_y = map_dict[f"max_{y_key}"]
    num_y = int((max_y - min_y) / dy + 1)

    def get_index(x: np.float64, y: np.float64) -> int:
        xgrid = round(x / dx) * dx
        ygrid = round(y / dx) * dy
        npoints = len(map_dict[x_key])  # TODO: fix
        map_index = int((xgrid - min_x) / dx * num_y + (ygrid - min_y) / dy)
        if map_index >= npoints:
            map_index = npoints - 1
        return map_index

    x1 = floor(x / dx) * dx
    y1 = floor(y / dy) * dy
    x2 = x1 + dx
    y2 = y1 + dy

    idx_11 = get_index(x1, y1)
    idx_21 = get_index(x2, y1)
    idx_12 = get_index(x1, y2)
    idx_22 = get_index(x2, y2)

    vals = map_dict[val_key]
    v11 = vals[idx_11]
    v21 = vals[idx_21]
    v12 = vals[idx_12]
    v22 = vals[idx_22]

    return (
        1
        / ((x2 - x1) * (y2 - y1))
        * (
            v11 * (x2 - x) * (y2 - y)
            + v21 * (x - x1) * (y2 - y)
            + v12 * (x2 - x) * (y - y1)
            + v22 * (x - x1) * (y - y1)
        )
    )


def interpolate3D(
    map_dict: dict,
    x: float,
    y: float,
    z: float,
    x_key: str = "x",
    y_key: str = "y",
    z_key: str = "z",
    val_key: str = "val",
    is_relative: bool = True,
) -> float:
    """Trilinear interpolation of a 3D map at position (x, y, z)."""
    if not is_relative:
        x -= map_dict["detector_position"][x_key]
        y -= map_dict["detector_position"][y_key]
        z -= map_dict["detector_position"][z_key]

    dx = map_dict[f"d{x_key}"]
    dy = map_dict[f"d{y_key}"]
    dz = map_dict[f"d{z_key}"]

    min_x = map_dict[f"min_{x_key}"]
    min_y = map_dict[f"min_{y_key}"]
    min_z = map_dict[f"min_{z_key}"]
    max_y = map_dict[f"max_{y_key}"]
    max_z = map_dict[f"max_{z_key}"]

    num_y = int((max_y - min_y) / dy + 1)
    num_z = int((max_z - min_z) / dz + 1)

    def get_index(xq: float, yq: float, zq: float) -> int:
        ix = round((xq - min_x) / dx)
        iy = round((yq - min_y) / dy)
        iz = round((zq - min_z) / dz)
        return ix * num_y * num_z + iy * num_z + iz

    # Surrounding grid corners
    x1 = floor((x - min_x) / dx) * dx + min_x
    y1 = floor((y - min_y) / dy) * dy + min_y
    z1 = floor((z - min_z) / dz) * dz + min_z
    x2 = x1 + dx
    y2 = y1 + dy
    z2 = z1 + dz

    idx_000 = get_index(x1, y1, z1)
    idx_100 = get_index(x2, y1, z1)
    idx_010 = get_index(x1, y2, z1)
    idx_110 = get_index(x2, y2, z1)
    idx_001 = get_index(x1, y1, z2)
    idx_101 = get_index(x2, y1, z2)
    idx_011 = get_index(x1, y2, z2)
    idx_111 = get_index(x2, y2, z2)

    vals = map_dict[val_key]
    v000 = vals[idx_000]
    v100 = vals[idx_100]
    v010 = vals[idx_010]
    v110 = vals[idx_110]
    v001 = vals[idx_001]
    v101 = vals[idx_101]
    v011 = vals[idx_011]
    v111 = vals[idx_111]

    xd = (x - x1) / (x2 - x1)
    yd = (y - y1) / (y2 - y1)
    zd = (z - z1) / (z2 - z1)

    # Trilinear interpolation formula
    c00 = v000 * (1 - xd) + v100 * xd
    c01 = v001 * (1 - xd) + v101 * xd
    c10 = v010 * (1 - xd) + v110 * xd
    c11 = v011 * (1 - xd) + v111 * xd

    c0 = c00 * (1 - yd) + c10 * yd
    c1 = c01 * (1 - yd) + c11 * yd

    return c0 * (1 - zd) + c1 * zd


def read_hpge_map(filename: str, field_name: str, hpge_position: list) -> dict:
    """Reads the HPGe map from the given file and returns a dictionary with the map data.

    Args:
        filename (str): Path to the file containing the HPGe map.
        field_name (str): Name of the field in the file to read.
        hpge (LogicalVolume): The HPGe detector volume.

    Returns
    -------
        dict: A dictionary containing the map data.
    """
    store = LH5Store()
    dt_data, _ = store.read(f"/{field_name}", filename)

    unit_conversion = {"um": 1e-6, "mm": 1e-3, "cm": 1e-2, "m": 1}
    coord_keys = {"x", "y", "z", "r"}

    result = {
        "detector_position": {"x": hpge_position[0], "y": hpge_position[1], "z": hpge_position[2]}
    }

    # Separate coordinates and value fields
    coords = {}
    values = {}

    for key in dt_data:
        data = dt_data[key].nda
        unit = dt_data[key].attrs.get("units", "m")
        data_m = data * unit_conversion.get(unit, 1)

        if key in coord_keys:
            coords[key] = data_m
        else:
            values[key] = data_m

    # Add coordinate arrays
    result.update(coords)

    # Add value arrays
    result.update(values)

    # Calculate metadata (e.g. grid spacing)
    for key, item in coords.items():
        unique_vals = np.unique(item)
        if len(unique_vals) > 1:
            result[f"d{key}"] = abs(unique_vals[1] - unique_vals[0])
            result[f"min_{key}"] = np.min(unique_vals)
            result[f"max_{key}"] = np.max(unique_vals)

    return result
