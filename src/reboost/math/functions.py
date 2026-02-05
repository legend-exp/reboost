from __future__ import annotations

import logging

import awkward as ak
import numpy as np
from lgdo import Array, VectorOfVectors
from scipy.optimize import brentq

from .. import units

log = logging.getLogger(__name__)


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

    results = ak.Array(results)

    return units.attach_units(results, "mm")


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


def ExLinT(distances: ak.Array, fccd: float, alpha: float, beta: float):
    r"""MGTDExLinT HPGe activeness model (Exponential + Linear) Here is how the function works.

    - If d<=0: 0
    - If 0< d < D: B * [exp(d/beta) -1]
    - If D <= d < FCCD: 1+ (d - FCCD) / alpha
    - If d >= FCCD: 1
    - if d is Nan: 1

    where D is the transition between switching from exponential model to the linear model. Yout get this point by solving this equation
         alpha + D - FCCD + beta * exp(-D/beta) - beta = 0
    You get this equation by imposing the continuity and differentiability at the transition point D

    Then, you also solve for B, which is given as:
         B = (beta/alpha) * exp(-D/beta)

    Notes: np.nan is assigned with the activeness = 1, matching the reboost convention

    Parameters to be fed in the function:
    distances: distance from the step to the surface
    fccd: full charge collection depth
    alpha: parameter for the linear model. 1 / alpha is the slope
    beta: parameter for the exponential model. Used in the exponent

    Returns
    -------
    activeness: :class::`VectorOfVectors` or :class:`Array` of the activeness
    """
    # Convert to ak
    """if isinstance(distances, LGDO):
        distances_ak = distances.view_as("ak")
    elif not isinstance(distances, ak.Array):
        distances_ak =ak.Array(distances)
    else:
        distances_ak = distances"""

    # Covert to ak
    distances_ak = units.units_conv_ak(distances, "mm")

    # Flatten the distance to 1D
    distances_flat = (
        ak.flatten(distances_ak).to_numpy() if distances_ak.ndim > 1 else distances_ak.to_numpy()
    )
    lengths = ak.num(distances_ak) if distances_ak.ndim > 1 else len(distances_ak)
    # --- parameter checks ----
    if fccd < 0:
        raise ValueError("FCCD must be >==0.")
    if alpha < 0 or alpha > fccd:
        raise ValueError("alpha must satisfy 0<= alpha <= FCCD.")
    if beta < 0:
        raise ValueError("beta must be >=0.")
    if beta * (1.0 - np.exp(-fccd / beta)) > alpha + 1e-15:
        raise ValueError(
            " Unphysical parameters: beta * (1-exp(-FCCD/beta)) must be <= alpha. This condition is needed to ensure the smooth transition from exp to linear fccd model transition."
        )

    if fccd == 0:
        results = np.full_like(distances_flat, np.nan, dtype=np.float64)
        mask_full = (distances_flat > 0) | np.isnan(distances_flat)
        results[mask_full] = 1.0
        results[~mask_full] = 0.0
        results = ak.unflatten(ak.Array(results), lengths) if distances_ak.ndim > 1 else results
        results = ak.Array(results)
        # return VectorOfVectors(results) if results.ndim > 1 else Array(results)
        return units.attach_units(results, "mm")

    if alpha == 0:
        results = np.full_like(distances_flat, np.nan, dtype=np.float64)
        mask_full = (distances_flat > fccd) | np.isnan(distances_flat)
        results[mask_full] = 1.0
        results[~mask_full] = 0.0
        results = ak.unflatten(ak.Array(results), lengths) if distances_ak.ndim > 1 else results
        results = ak.Array(results)

        return units.attach_units(results, "mm")

    if beta == 0:
        results = np.full_like(distances_flat, np.nan, dtype=np.float64)
        mask_nan = np.isnan(distances_flat)
        mask_full = (distances_flat >= fccd) & (~mask_nan)
        mask_zero = (distances_flat <= 0.0) & (~mask_nan)
        mask_lin = ~(mask_nan | mask_full | mask_zero)

        results[mask_nan] = 1.0
        results[mask_full] = 1.0
        results[mask_zero] = 0.0
        results[mask_lin] = distances_flat[mask_lin] / fccd
        results = ak.unflatten(ak.Array(results), lengths) if distances_ak.ndim > 1 else results
        results = ak.Array(results)
        return units.attach_units(results, "mm")

    def f(D):
        return alpha + D - fccd + beta * np.exp(-D / beta) - beta

    lo = max(0.0, fccd - alpha)
    hi = fccd

    # using brentq solver. #Matching this solver with the one used by the Majorana
    D = brentq(f, lo, hi)

    # Compute B
    B = (beta / alpha) * np.exp(-D / beta)

    results = np.full_like(distances_flat, np.nan, dtype=np.float64)

    mask_nan = np.isnan(distances_flat)
    mask_full = (distances_flat >= fccd) & (~mask_nan)
    mask_zero = (distances_flat <= 0) & (~mask_nan)
    mask_lin = (distances_flat >= D) & (distances_flat < fccd) & (~mask_nan)
    mask_exp = (distances_flat > 0.0) & (distances_flat < D) & (~mask_nan)

    results[mask_nan] = 1.0
    results[mask_full] = 1.0
    results[mask_zero] = 0.0
    results[mask_lin] = 1.0 + (distances_flat[mask_lin] - fccd) / alpha
    results[mask_exp] = B * (-1.0 + np.exp(distances_flat[mask_exp] / beta))

    results = ak.unflatten(ak.Array(results), lengths) if distances_ak.ndim > 1 else results
    results = ak.Array(results)

    return units.attach_units(results, "mm")
