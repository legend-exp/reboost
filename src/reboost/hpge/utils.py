from __future__ import annotations

from math import floor

import awkward as ak
import numpy as np
from lgdo.lh5 import LH5Store


def interpolate2D(
    map_dict: dict,
    x: np.float64,
    y: np.float64,
    x_key: str = "x",
    y_key: str = "y",
    val_key: str = "val",
) -> np.float64:
    """Bilinear interpolation of a 2D map at position (x, y).

    Parameters
    ----------
        map_dict : dict
            Dictionary containing the map data.
        x : np.float64
            X-coordinate for interpolation.
        y : np.float64
            Y-coordinate for interpolation.
        x_key : str
            Key for the x-coordinate in the map dictionary.
        y_key : str
            Key for the y-coordinate in the map dictionary.
        val_key : str
            Key for the value to be interpolated in the map dictionary.

    Returns
    -------
        np.float64
            Interpolated value at (x, y).

    Notes
    -----
        - The function assumes that the map_dict contains the necessary keys for
          x, y, and value arrays.
        - The function uses bilinear interpolation to compute the value at the
          specified coordinates.
        - The function returns the interpolated value as a float.
    """
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
    x: np.float64,
    y: np.float64,
    z: np.float64,
    x_key: str = "x",
    y_key: str = "y",
    z_key: str = "z",
    val_key: str = "val",
) -> np.float64:
    """Trilinear interpolation of a 3D map at position (x, y, z).

    Parameters
    ----------
        map_dict : dict
            Dictionary containing the map data.
        x : np.float64
            X-coordinate for interpolation.
        y : np.float64
            Y-coordinate for interpolation.
        z : np.float64
            Z-coordinate for interpolation.
        x_key : str
            Key for the x-coordinate in the map dictionary.
        y_key : str
            Key for the y-coordinate in the map dictionary.
        z_key : str
            Key for the z-coordinate in the map dictionary.
        val_key : str
            Key for the value to be interpolated in the map dictionary.

    Returns
    -------
        np.float64
            Interpolated value at (x, y, z).

    Notes
    -----
        - The function assumes that the map_dict contains the necessary keys for
          x, y, z, and value arrays.
        - The function uses trilinear interpolation to compute the value at the
          specified coordinates.
        - The function returns the interpolated value as a np.float64.
    """
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


def get_relative_loc(
    xloc: ak.Array, yloc: ak.Array, zloc: ak.Array, detector_position: list
) -> ak.Array:
    return xloc - detector_position[0], yloc - detector_position[1], zloc - detector_position[2]


def read_hpge_map(filename: str, field_name: str) -> dict:
    """Reads a HPGe map from an lh5 file and returns the data as a dictionary.

    Parameters
    ----------
        filename : str
            Path to the file containing the HPGe map.
        field_name : str
            Name of the field to be read from the file.

    Returns
    -------
        dict
            Dictionary containing the map data.

    Notes
    -----
        - The function uses the LH5Store class to read the data from the file.
        - Assumes the data is stored in a specific format with coordinate and value arrays.
        - The function returns a dictionary with coordinate arrays and value arrays.
    """
    store = LH5Store()
    dt_data, _ = store.read(f"/{field_name}", filename)

    unit_conversion = {"um": 1e-6, "mm": 1e-3, "cm": 1e-2, "m": 1}
    coord_keys = {"x", "y", "z", "r"}

    result = {}
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
