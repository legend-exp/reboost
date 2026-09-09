from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import NamedTuple

import lgdo
import lh5
import numpy as np
import pint
from dbetto import AttrsDict
from numpy.typing import DTypeLike
from scipy.interpolate import RegularGridInterpolator


class HPGePulseShapeLibrary(NamedTuple):
    """A set of templates defined in the cylindrical-like (r, z) HPGe plane."""

    waveforms: np.ndarray
    "Field, function of the coordinates (r, z)."
    r_units: pint.Unit
    "Physical units of the coordinate `r`."
    z_units: pint.Unit
    "Physical units of the coordinate `z`."
    t_units: pint.Unit
    "Physical units of the times."
    r: np.ndarray
    "One dimensional arrays specifying the radial coordinates"
    z: np.ndarray
    "One dimensional arrays specifying the z coordinates"
    t: np.ndarray
    "Times used to define the waveforms"


def load_hpge_pulse_shape_library(
    filename: str,
    obj: str,
    field: str,
    out_of_bounds_val: float = np.nan,
    dtype: DTypeLike | None = None,
) -> HPGePulseShapeLibrary:
    """Create the pulse shape library, holding simulated waveforms.

    Reads from disk the following data structure: ::

        FILENAME/
        └── OBJ · struct{r,z,dt,t0,FIELD}
            ├── r · array<1>{real} ── {'units': 'UNITS'}
            ├── z · array<1>{real} ── {'units': 'UNITS'}
            ├── dt · real ── {'units': 'UNITS'}
            ├── t0 · real ── {'units': 'UNITS'}
            └── FIELD · array<3>{real} ── {'units': 'UNITS'}

    The conventions follow those used for :func:`load_hpge_rz_field`.
    For the FIELD the first and second dimensions are `r` and `z`, respectively, with the last
    dimension representing the waveform. dt and t0 define the timestamps for the waveforms.


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
    dtype
        if not ``None``, the waveforms are cast to this data type after reading.
        ``float32`` halves the memory taken by the library and is enough for a
        A/E estimate at the percent level.
    """
    data = lh5.read(obj, filename)

    if not isinstance(data, lgdo.Struct):
        msg = f"{obj} in {filename} is not an LGDO Struct"
        raise TypeError(msg)

    t0 = data["t0"].value
    dt = data["dt"].value

    t0_u = data["t0"].attrs["units"]
    dt_u = data["dt"].attrs["units"]

    if t0_u != dt_u:
        msg = "t0 and dt must have the same units"
        raise ValueError(msg)

    tu = t0_u

    data = AttrsDict(
        {
            k: np.nan_to_num(data[k].view_as("np", with_units=(k != field)), nan=out_of_bounds_val)
            for k in ("r", "z", field)
        }
    )

    times = t0 + np.arange(np.shape(data[field])[2]) * dt

    waveforms = data[field] if dtype is None else np.asarray(data[field], dtype=dtype)

    return HPGePulseShapeLibrary(waveforms, data.r.u, data.z.u, tu, data.r.m, data.z.m, times)


def load_hpge_pulse_shape_libraries(
    filename: str,
    obj: str,
    angles: Sequence[int] = (0,),
    out_of_bounds_val: float = np.nan,
    dtype: DTypeLike | None = None,
) -> dict[int, HPGePulseShapeLibrary]:
    """Read the pulse shape libraries of a detector for several crystal-axis angles.

    The drift velocity in germanium depends on the orientation of the crystal
    axes, so the waveforms are simulated on the `(r, z)` grid separately for a
    few azimuthal angles. They are stored in the same LH5 struct, in the fields
    ``waveform_000_deg``, ``waveform_045_deg`` and so on, with the angle in
    degrees padded to three digits.

    Each library is read by :func:`load_hpge_pulse_shape_library`. The returned
    mapping is keyed by the angle in degrees.

    Parameters
    ----------
    filename
        name of the LH5 file containing the libraries.
    obj
        name of the HDF5 dataset where the data is saved, usually the detector
        name.
    angles
        crystal-axis angles, in degrees, to read. The waveforms dominate the
        size of the file (tens of gigabytes for a full detector), so only the
        angles that are actually used should be read.
    out_of_bounds_val
        value to use to replace NaNs in the waveform values.
    dtype
        if not ``None``, the waveforms are cast to this data type after reading.
    """
    return {
        angle: load_hpge_pulse_shape_library(
            filename,
            obj,
            f"waveform_{angle:03d}_deg",
            out_of_bounds_val=out_of_bounds_val,
            dtype=dtype,
        )
        for angle in angles
    }


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


def load_hpge_rz_field(
    filename: str, obj: str, field: str, out_of_bounds_val: float = np.nan, **kwargs
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

    Before returning a :class:`~reboost.hpge.utils.HPGeRZField`, the gridded field is fed to
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
        raise TypeError(msg)

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


def load_hpge_drift_time_maps(
    filename: str,
    obj: str,
    angles: Sequence[int] = (0, 45),
    out_of_bounds_val: float = np.nan,
    **kwargs,
) -> dict[int, HPGeRZField]:
    """Read the drift-time maps of a detector for several crystal-axis angles.

    The drift velocity in germanium depends on the orientation of the crystal
    axes, so the drift time is mapped on the `(r, z)` grid separately for a few
    azimuthal angles. The maps are stored in the same LH5 struct, in the fields
    ``drift_time_000_deg``, ``drift_time_045_deg`` and so on, with the angle in
    degrees padded to three digits.

    Each map is read by :func:`load_hpge_rz_field`. The returned mapping is keyed
    by the angle in degrees and can be passed to
    :func:`reboost.hpge.drift_time_crystal_axes`.

    Parameters
    ----------
    filename
        name of the LH5 file containing the maps.
    obj
        name of the HDF5 dataset where the data is saved, usually the detector
        name.
    angles
        crystal-axis angles, in degrees, to read.
    out_of_bounds_val
        value to use to replace NaNs in the field values.
    **kwargs
        further keyword arguments forwarded to :func:`load_hpge_rz_field`, and
        from there to :class:`scipy.interpolate.RegularGridInterpolator`.
    """
    return {
        angle: load_hpge_rz_field(
            filename,
            obj,
            f"drift_time_{angle:03d}_deg",
            out_of_bounds_val=out_of_bounds_val,
            **kwargs,
        )
        for angle in angles
    }
