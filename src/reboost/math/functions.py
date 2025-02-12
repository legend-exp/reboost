from __future__ import annotations

import logging

import awkward as ak
import numpy as np
from lgdo import Array, VectorOfVectors
from lgdo.types import LGDO

log = logging.getLogger(__name__)


def piecewise_linear_activeness(
    distances: VectorOfVectors | ak.Array, fccd: float, tl: float
) -> VectorOfVectors | Array:
    r"""Piecewise linear HPGe activeness model.

    Based on:

    .. math::

        f(d) =
        \begin{cases}
        0 & \text{if } d < t, \\
        \frac{x-l}{f - l} & \text{if } t \leq d < f, \\
        1 & \text{otherwise.}
        \end{cases}

    Where:
    - `d`: Distance to surface,
    - `l`: Depth of transition layer start
    - `f`: Full charge collection depth (FCCD).

    In addition, any distance of `np.nan` (for example if the calculation
    was not performed for some steps) is assigned an activeness of one.

    Parameters
    ----------
    distances
        the distance from each step to the detector surface. Can be either a
        `numpy` or `awkward` array, or a LGDO `VectorOfVectors` or `Array`. The computation
        is performed for each element and the shape preserved in the output.

    fccd
        the value of the FCCD
    tl
        the start of the transition layer.

    Returns
    -------
    a :class:`VectorOfVectors` or :class:`Array` of the activeness
    """
    # convert to ak
    if isinstance(distances, LGDO):
        distances_ak = distances.view_as("ak")
    elif not isinstance(distances, ak.Array):
        distances_ak = ak.Array(distances)
    else:
        distances_ak = distances

    # compute the linear piecewise
    results = ak.where(
        (distances_ak > fccd) | np.isnan(distances_ak),
        1,
        ak.where(distances_ak <= tl, 0, (distances_ak - tl) / (fccd - tl)),
    )
    return VectorOfVectors(results) if results.ndim > 1 else Array(results.to_numpy())
