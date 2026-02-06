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
        └── OBJ · struct{r,z,dt,t0,FIELD}
            ├── r · array<1>{real} ── {'units': 'UNITS'}
            ├── z · array<1>{real} ── {'units': 'UNITS'}
            ├── phi · array<1>{real} ── {'units': 'deg'} (optional)
            ├── dt · real ── {'units': 'UNITS'}
            ├── t0 · real ── {'units': 'UNITS'}
            └── FIELD · array<3 or 4>{real} ── {'units': 'UNITS'}

    The conventions follow those used for :func:`get_hpge_rz_field`.
    For the FIELD the first and second dimensions are `r` and `z`, respectively, with the last
    dimension representing the waveform. If phi is present, the third dimension is phi and the
    fourth is the waveform. dt and t0 define the timestamps for the waveforms.


    Parameters
    ----------
    filename
        name of the LH5 file containing the gridded scalar field.
    obj
        name of the HDF5 dataset where the data is saved.
    field
        name of the HDF5 dataset holding the waveforms.
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

    if "units" not in data[field].attrs:
        data[field].attrs["units"] = ""

    tu = t0_u

    # Check if phi is present
    has_phi = "phi" in data
    keys_to_read = ["r", "z", field]
    if has_phi:
        keys_to_read.append("phi")

    data = AttrsDict(
        {
            k: np.nan_to_num(data[k].view_as("np", with_units=True), nan=out_of_bounds_val)
            for k in keys_to_read
        }
    )

    # Extract time dimension based on whether phi is present
    time_dim_idx = 3 if has_phi else 2
    times = t0 + np.arange(np.shape(data[field].m)[time_dim_idx]) * dt

    phi_array = data.phi.m if has_phi else None

    return HPGePulseShapeLibrary(
        data[field].m, data.r.u, data.z.u, tu, data.r.m, data.z.m, times, phi_array
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
