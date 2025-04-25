from __future__ import annotations

from typing import Callable, NamedTuple

import lgdo
import pint
from dbetto import AttrsDict
from lgdo import lh5
from scipy.interpolate import RegularGridInterpolator


class HPGeScalarRZField(NamedTuple):
    """A scalar field defined in the cylindrical-like (r, z) HPGe plane."""

    φ: Callable
    "Scalar field, function of the coordinates (r, z)."
    r_units: pint.Unit
    "Physical units of the coordinate `r`."
    z_units: pint.Unit
    "Physical units of the coordinate `z`."
    values_units: pint.Unit
    "Physical units of the field."


def get_hpge_scalar_rz_field(filename: str, obj: str, field: str) -> HPGeScalarRZField:
    """Create an interpolator for a gridded scalar HPGe field defined on (r, z).

    Reads from disk the following data structure: ::

        FILENAME/
        └── OBJ · struct{r,z,FIELD}
            ├── r · array<1>{real}
            ├── z · array<1>{real}
            └── FIELD · array<2>{real}

    where ``FILENAME``, ``OBJ`` and ``FIELD`` are provided as
    arguments to this function. `obj` is a :class:`~lgdo.types.struct.Struct`,
    `r` and `z` are one dimensional arrays specifying the radial and z
    coordinates of the rectangular grid — not the coordinates of each single
    grid point. `field` is instead a two-dimensional array specifying the field
    value at each grid point. The first and second dimensions are `r` and `z`,
    respectively.

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
    """
    data = lh5.read(obj, filename)

    if not isinstance(data, lgdo.Struct):
        msg = f"{obj} in {filename} is not an LGDO Struct"
        raise ValueError(msg)

    data = AttrsDict({k: data[k].view_as("np", with_units=True) for k in ("r", "z", field)})

    interpolator = RegularGridInterpolator((data.r.m, data.z.m), data[field].m)

    return HPGeScalarRZField(interpolator, data.r.u, data.z.u, data[field].u)
