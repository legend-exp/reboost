from __future__ import annotations

import logging
import math
import sys

import awkward as ak
import numpy as np
from lgdo import Array, VectorOfVectors

from .. import units

log = logging.getLogger(__name__)


def _get_transition_point(
    fccd_in_mm: float, alpha_in_mm: float, beta_in_mm: float, n: int = 200
) -> float:
    """Get the transition point for the exlint model."""

    def f(dist):
        return (
            alpha_in_mm + dist - fccd_in_mm + beta_in_mm * math.exp(-dist / beta_in_mm) - beta_in_mm
        )

    lo = fccd_in_mm - alpha_in_mm
    hi = fccd_in_mm

    for _ in range(n):
        mid = 0.5 * (lo + hi)
        if f(lo) * f(mid) <= 0:
            hi = mid
        else:
            lo = mid

    return 0.5 * (lo + hi)


def ex_lint_activeness(
    distances: ak.Array, fccd_in_mm: float, alpha_in_mm: float, beta_in_mm: float
) -> ak.Array:
    r"""Exponentially modified linear (exlint) HPGe activeness model.

    The model consists of a exponential component (with exponent distance/beta) near the surface,
    transitioning to a linear component (with slope alpha) closer to the FCCD.

    .. math::

        f(d) =
        \begin{cases}
        B\times e^{d/\beta} & \text{if } d < T, \\
        \frac{d+\alpha - f}{\alpha} & \text{if } d<f, \\
        1 & \text{otherwise.}
        \end{cases}

    - `d`: Distance to surface,
    - `T`: transition from exponential to linear activeness.
    - `beta`: exponential rise parameter.
    - `alpha`: linear rise parameter.
    - `f`: Full charge collection depth (FCCD).
    - `B`: normalisation of the exponential component.

    Parameters
    ----------
    distances
        the distance from each step to the detector surface. The computation
        is performed for each element and the shape preserved in the output.
    fccd_in_mm
        the value of the FCCD
    alpha_in_mm
        the alpha parameter of the TL model.
    beta_in_mm
        the beta parameter of the TL model.
    """
    distances_ak = units.units_conv_ak(distances, "mm")

    # Bounds checks
    if fccd_in_mm < 0.0:
        msg = "FCCD must be positive"
        raise ValueError(msg)

    if (alpha_in_mm < 0.0) or alpha_in_mm > fccd_in_mm:
        msg = "Alpha must be positive and less than FCCD"
        raise ValueError(msg)

    if (beta_in_mm < 0.0) or (
        (beta_in_mm > 0) and (beta_in_mm * (1 - math.exp(-fccd_in_mm / beta_in_mm)) > alpha_in_mm)
    ):
        msg = "Beta must be positive and satisfy beta*(1-exp(-FCCD/beta)) < alpha"
        raise ValueError(msg)

    # Defaults
    if fccd_in_mm == 0:
        trans_pt = 0.0
        B = 0.0
    elif (beta_in_mm == 0) or (fccd_in_mm / beta_in_mm > math.log(sys.float_info.max)):
        trans_pt = fccd_in_mm - alpha_in_mm
        B = 0.0
    else:
        trans_pt = _get_transition_point(fccd_in_mm, alpha_in_mm, beta_in_mm)
        B = beta_in_mm / alpha_in_mm * math.exp(-trans_pt / beta_in_mm)

    # now run the computation
    distances_flat = (
        ak.flatten(distances_ak).to_numpy() if distances_ak.ndim > 1 else distances_ak.to_numpy()
    )

    # compute the activeness
    results = np.full_like(distances_flat, np.nan, dtype=np.float64)
    lengths = ak.num(distances_ak) if distances_ak.ndim > 1 else len(distances_ak)

    mask1 = (distances_flat > fccd_in_mm) | np.isnan(distances_flat)
    mask2 = (distances_flat >= trans_pt) & (~mask1)
    mask3 = (distances_flat < trans_pt) & (beta_in_mm == 0)
    mask4 = ~(mask1 | mask2 | mask3)

    # assign the values
    results[mask1] = 1
    results[mask2] = 1.0 + (distances_flat[mask2] - fccd_in_mm) / (
        alpha_in_mm + sys.float_info.epsilon
    )
    results[mask3] = 0.0
    results[mask4] = B * (
        -1.0 + np.exp(distances_flat[mask4] / (beta_in_mm + sys.float_info.epsilon))
    )

    # reshape
    results = ak.unflatten(ak.Array(results), lengths) if distances_ak.ndim > 1 else results
    return ak.Array(results)


def piecewise_linear_activeness(distances: ak.Array, fccd_in_mm: float, dlf: float) -> ak.Array:
    r"""Piecewise linear HPGe activeness model.

    Based on:

    .. math::

        f(d) =
        \begin{cases}
        0 & \text{if } d < f*l, \\
        \frac{x-f*l}{f - f*l} & \text{if } t \leq d < f, \\
        1 & \text{otherwise.}
        \end{cases}

    Where:

    - `d`: Distance to surface,
    - `l`: Dead layer fraction, the fraction of the FCCD which is fully inactive
    - `f`: Full charge collection depth (FCCD).

    In addition, any distance of `np.nan` (for example if the calculation
    was not performed for some steps) is assigned an activeness of one.

    Parameters
    ----------
    distances
        the distance from each step to the detector surface. The computation
        is performed for each element and the shape preserved in the output.

    fccd_in_mm
        the value of the FCCD
    dlf
        the fraction of the FCCD which is fully inactive.

    Returns
    -------
    a :class:`VectorOfVectors` or :class:`Array` of the activeness
    """
    # convert to ak
    distances_ak = units.units_conv_ak(distances, "mm")

    dl = fccd_in_mm * dlf
    distances_flat = (
        ak.flatten(distances_ak).to_numpy() if distances_ak.ndim > 1 else distances_ak.to_numpy()
    )

    # compute the linear piecewise
    results = np.full_like(distances_flat, np.nan, dtype=np.float64)
    lengths = ak.num(distances_ak) if distances_ak.ndim > 1 else len(distances_ak)

    mask1 = (distances_flat > fccd_in_mm) | np.isnan(distances_flat)
    mask2 = (distances_flat <= dl) & (~mask1)
    mask3 = ~(mask1 | mask2)

    # assign the values
    results[mask1] = 1
    results[mask2] = 0
    results[mask3] = (distances_flat[mask3] - dl) / (fccd_in_mm - dl)

    # reshape
    results = ak.unflatten(ak.Array(results), lengths) if distances_ak.ndim > 1 else results

    return ak.Array(results)


def vectorised_active_energy(
    distances: ak.Array,
    edep: ak.Array,
    fccd: float | list,
    dlf: float | list,
) -> VectorOfVectors | Array:
    r"""Energy after piecewise linear HPGe activeness model vectorised over FCCD or dead layer fraction.

    Based on the same linear activeness function as :func:`piecewise_linear_activeness`. However,
    this function vectorises the calculation to provide a range of output energies varying the fccd or
    dead layer fraction. Either fccd or dlf can be a list. This adds an extra dimension to the
    output, with the same length as the input fccd or dlf list.

    .. warning:
        It is not currently implemented to vary both dlf and fccd.

    Parameters
    ----------
    distances
        the distance from each step to the detector surface. Can be either a
        `awkward` array, or a LGDO `VectorOfVectors` . The computation
        is performed for each element and the first dimension is preserved, a
        new dimension is added vectorising over the FCCD or DLF.
    edep
        the energy for each step.
    fccd
        the value of the FCCD, can be a list.
    dlf
        the fraction of the FCCD which is fully inactive, can be a list.

    Returns
    -------
    Activeness for each set of parameters
    """
    # add checks on fccd, dlf
    fccd = np.array(fccd)
    dlf = np.array(dlf)

    if (fccd.ndim + dlf.ndim) > 1:
        msg = "Currently only one of FCCD and dlf can be varied"
        raise NotImplementedError(msg)

    # convert fccd and or dlf to the right shape
    if fccd.ndim == 0:
        if dlf.ndim == 0:
            dlf = dlf[np.newaxis]
        fccd = np.full_like(dlf, fccd)

    dl = fccd * dlf

    def _convert(field, unit):
        # convert to ak
        field_ak = units.units_conv_ak(field, unit)

        return field_ak, ak.flatten(field_ak).to_numpy()[:, np.newaxis]

    distances_ak, distances_flat = _convert(distances, "mm")
    _, edep_flat = _convert(edep, "keV")
    runs = ak.num(distances_ak, axis=-1)

    # vectorise fccd or tl

    fccd_list = np.tile(fccd, (len(distances_flat), 1))
    dl_list = np.tile(dl, (len(distances_flat), 1))
    distances_shaped = np.tile(distances_flat, (1, len(dl)))

    # compute the linear piecewise
    results = np.full_like(fccd_list, np.nan, dtype=np.float64)

    # Masks
    mask1 = (distances_shaped > fccd_list) | np.isnan(distances_shaped)
    mask2 = ((distances_shaped <= dl_list) | (fccd_list == dl_list)) & ~mask1
    mask3 = ~(mask1 | mask2)  # Safe, avoids recomputing anything expensive

    # Assign values
    results[mask1] = 1.0
    results[mask2] = 0.0
    results[mask3] = (distances_shaped[mask3] - dl_list[mask3]) / (
        fccd_list[mask3] - dl_list[mask3]
    )

    energy = ak.sum(ak.unflatten(results * edep_flat, runs), axis=-2)

    return units.attach_units(energy, "keV")
