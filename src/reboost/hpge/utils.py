from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import lgdo
import numpy as np
import pint
from dbetto import AttrsDict
from lgdo import lh5
from scipy.interpolate import RegularGridInterpolator


class HPGePulseShapeLibrary(NamedTuple):
    """A set of templates defined in the cylindrical-like (r, z) HPGe plane."""

    waveforms: np.array
    "Field, function of the coordinates (r, z) or (r, z, phi)."
    r_units: pint.Unit
    "Physical units of the coordinate `r`."
    z_units: pint.Unit
    "Physical units of the coordinate `z`."
    t_units: pint.Unit
    "Physical units of the times."
    r: np.array
    "One dimensional arrays specifying the radial coordinates"
    z: np.array
    "One dimensional arrays specifying the z coordinates"
    t: np.array
    "Times used to define the waveforms"
    phi: np.array | None = None
    "One dimensional array specifying the phi angles in degrees (optional)"


def get_hpge_pulse_shape_library(
    filename: str, obj: str, field: str, out_of_bounds_val: int | float = np.nan
) -> HPGePulseShapeLibrary:
    """Create the pulse shape library, holding simulated waveforms.

    Reads from disk the following data structure: ::

        FILENAME/
        └── OBJ · struct{r,z,dt,t0,FIELD[,FIELD_phi1,FIELD_phi2,...]}
            ├── r · array<1>{real} ── {'units': 'UNITS'}
            ├── z · array<1>{real} ── {'units': 'UNITS'}
            ├── dt · real ── {'units': 'UNITS'}
            ├── t0 · real ── {'units': 'UNITS'}
            ├── FIELD · array<3>{real} ── {'units': 'UNITS'}
            ├── FIELD_phi1 · array<3>{real} ── {'units': 'UNITS'} (optional)
            └── FIELD_phi2 · array<3>{real} ── {'units': 'UNITS'} (optional)

    The conventions follow those used for :func:`get_hpge_rz_field`.
    For the FIELD the first and second dimensions are `r` and `z`, respectively, with the last
    dimension representing the waveform. dt and t0 define the timestamps for the waveforms.

    For phi-dependent pulse shape libraries, multiple fields can be provided with suffixes
    indicating the phi angle (e.g., ``waveforms_0deg``, ``waveforms_45deg``). These will be
    automatically detected and stacked into a 4D array internally.


    Parameters
    ----------
    filename
        name of the LH5 file containing the gridded scalar field.
    obj
        name of the HDF5 dataset where the data is saved.
    field
        name of the HDF5 dataset holding the waveforms. If phi-dependent fields exist,
        they should be named as ``{field}_{phi}deg`` (e.g., ``waveforms_0deg``, ``waveforms_45deg``).
    out_of_bounds_val
        value to use to replace NaNs in the field values.
    """
    data = lh5.read(obj, filename)

    if not isinstance(data, lgdo.Struct):
        msg = f"{obj} in {filename} is not an LGDO Struct"
        raise ValueError(msg)

    if "t0" in data:
        t0 = data["t0"].value
        t0_u = data["t0"].attrs["units"]
    else:
        t0 = 0
        t0_u = "ns"

    dt = data["dt"].value
    dt_u = data["dt"].attrs["units"]

    if (t0_u != dt_u) and (t0 != 0):
        msg = "t0 and dt must have the same units"
        raise ValueError(msg)

    # Check for phi-dependent fields (e.g., waveforms_0deg, waveforms_45deg)
    phi_fields = []
    phi_angles = []
    for key in data.keys():
        if key.startswith(f"{field}_") and key.endswith("deg"):
            # Extract phi angle from field name
            try:
                phi_str = key[len(field) + 1 : -3]  # Remove prefix and "deg" suffix
                phi_val = float(phi_str)
                phi_fields.append(key)
                phi_angles.append(phi_val)
            except ValueError:
                continue

    # Sort by phi angle
    if phi_fields:
        sorted_indices = np.argsort(phi_angles)
        phi_fields = [phi_fields[i] for i in sorted_indices]
        phi_angles = [phi_angles[i] for i in sorted_indices]

    tu = t0_u

    # Read the base field or phi-dependent fields
    if phi_fields:
        # Read phi-dependent fields and stack them
        waveform_list = []
        for phi_field in phi_fields:
            if "units" not in data[phi_field].attrs:
                data[phi_field].attrs["units"] = ""
            wf = np.nan_to_num(
                data[phi_field].view_as("np", with_units=True), nan=out_of_bounds_val
            )
            waveform_list.append(wf.m)

        # Stack into 4D array (r, z, phi, time)
        waveforms_4d = np.stack(waveform_list, axis=2)
        phi_array = np.array(phi_angles)
    else:
        # No phi-dependent fields, use the base field
        if "units" not in data[field].attrs:
            data[field].attrs["units"] = ""
        wf = np.nan_to_num(data[field].view_as("np", with_units=True), nan=out_of_bounds_val)
        waveforms_4d = wf.m
        phi_array = None

    # Read r and z grids
    r_data = np.nan_to_num(data["r"].view_as("np", with_units=True), nan=out_of_bounds_val)
    z_data = np.nan_to_num(data["z"].view_as("np", with_units=True), nan=out_of_bounds_val)

    # Extract time dimension based on whether phi is present
    time_dim_idx = 3 if phi_array is not None else 2
    times = t0 + np.arange(np.shape(waveforms_4d)[time_dim_idx]) * dt

    return HPGePulseShapeLibrary(
        waveforms_4d, r_data.u, z_data.u, tu, r_data.m, z_data.m, times, phi_array
    )


class HPGeRZField(NamedTuple):
    """A field defined in the cylindrical-like (r, z) HPGe plane."""

    φ: Callable
    "Field, function of the coordinates (r, z)."
    r_units: pint.Unit
    "Physical units of the coordinate `r`."
    z_units: pint.Unit
    "Physical units of the coordinate `z`."
    φ_units: pint.Unit
    "Physical units of the field."
    ndim: int
    "Number of dimensions for the field"


def get_hpge_rz_field(
    filename: str, obj: str, field: str, out_of_bounds_val: int | float = np.nan, **kwargs
) -> HPGeRZField:
    """Create an interpolator for a gridded HPGe field defined on `(r, z)`.

    Reads from disk the following data structure: ::

        FILENAME/
        └── OBJ · struct{r,z,FIELD}
            ├── r · array<1>{real} ── {'units': 'UNITS'}
            ├── z · array<1>{real} ── {'units': 'UNITS'}
            └── FIELD · array<N+2>{real} ── {'units': 'UNITS'}

    where ``FILENAME``, ``OBJ`` and ``FIELD`` are provided as
    arguments to this function. `obj` is a :class:`~lgdo.types.struct.Struct`,
    `r` and `z` are one dimensional arrays specifying the radial and z
    coordinates of the rectangular grid — not the coordinates of each single
    grid point. In this coordinate system, the center of the p+ contact surface
    is at `(0, 0)`, with the p+ contact facing downwards. `field` is instead a
    ndim plus two-dimensional array specifying the field value at each grid point. The
    first and second dimensions are `r` and `z`, respectively, with the latter dimensions
    representing the dimensions of the output field.

    NaN values are interpreted as points outside the detector profile in the `(r, z)` plane.

    Before returning a :class:`HPGeScalarRZField`, the gridded field is fed to
    :class:`scipy.interpolate.RegularGridInterpolator`.

    Parameters
    ----------
    filename
        name of the LH5 file containing the gridded scalar field.
    obj
        name of the HDF5 dataset where the data is saved.
    field
        name of the HDF5 dataset holding the field values.
    out_of_bounds_val
        value to use to replace NaNs in the field values.
    """
    data = lh5.read(obj, filename)

    if not isinstance(data, lgdo.Struct):
        msg = f"{obj} in {filename} is not an LGDO Struct"
        raise ValueError(msg)

    data = AttrsDict(
        {
            k: np.nan_to_num(data[k].view_as("np", with_units=True), nan=out_of_bounds_val)
            for k in ("r", "z", field)
        }
    )
    ndim = data[field].m.ndim - 2
    interpolator = RegularGridInterpolator(
        (data.r.m, data.z.m), data[field].m, **(kwargs | {"fill_value": out_of_bounds_val})
    )

    return HPGeRZField(interpolator, data.r.u, data.z.u, data[field].u, ndim)
