from __future__ import annotations

import logging
import math
import sys

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


def ex_lin_activeness(distances: ak.Array, fccd: float, alpha: float, beta: float):
    r"""MGTDExLinT HPGe activeness model (Exponential + Linear).

    .. math::

        f(d) =
        \begin{cases}
        \mathrm{exp\_norm} * \left(e^{d/\beta} - 1\right) & \text{if } 0 \leq d < \mathrm{trans\_pt}, \\
        1 + \frac{d - f}{\alpha} & \text{if } \mathrm{trans\_pt} \leq d \leq f, \\
        1 & \text{if } d > f
        \end{cases}

    Where:

    - `d`: Distance to surface,
    - `f`: Full charge collection depth (FCCD).
    - `alpha`: the slope of the linear part of the function, which controls how quickly the activeness increases in the linear region. A smaller alpha results in a steeper increase, while a larger alpha results in a more gradual increase.
    - `beta`: the characteristic length scale of the exponential part of the function, which controls how quickly the activeness increases in the exponential region. A smaller beta results in a steeper increase, while a larger beta results in a more gradual increase.
    - `trans_pt`: the transition point between the exponential and linear parts of the function, which is determined by the parameters fccd, alpha, and beta. It is calculated by matching the functions and the derivatives at the transition point, which ensures a smooth transition between the two regions. The transition point is found by solving the equation:
         alpha + trans_pt - fccd + beta * exp(-trans_pt / beta) - beta = 0
    - `exp_norm`: the normalization factor for the exponential part of the function, which is determined by the parameters alpha and beta. It is calculated by ensuring that the exponential part of the function matches the linear part at the transition point, which ensures a smooth transition between the two regions.
            exp_norm = (beta / alpha) * exp(-trans_pt/beta)

    Parameters
    ----------
    distances
        the distance from each step to the detector surface. The computation
        is performed for each element and the shape preserved in the output.

    fccd_in_mm
        the value of the FCCD
    alpha
        the slope parameter for the linear part of the function, which controls how quickly the activeness increases in the linear region. 1 / alpha is the slope of the linear part of the function.
    beta
        the characteristic length scale for the exponential part of the function, which controls how quickly the activeness increases in the exponential region.

    Returns
    -------
    activeness: :class::`ak.Array` of the activeness per step
    """
    # Convert to ak
    distances_ak = units.units_conv_ak(distances, "mm")

    # Flatten the distance to 1D
    distances_flat = (
        ak.flatten(distances_ak).to_numpy() if distances_ak.ndim > 1 else distances_ak.to_numpy()
    )
    lengths = ak.num(distances_ak) if distances_ak.ndim > 1 else len(distances_ak)
    # --- parameter checks ----
    if fccd < 0:
        msg = "FCCD must be >=0."
        raise ValueError(msg)
    if alpha < 0 or alpha > fccd:
        msg = "alpha must satisfy 0<= alpha <= FCCD."
        raise ValueError(msg)
    if beta < 0:
        msg = "beta must be >=0."
        raise ValueError(msg)

    cond_check = beta * (1.0 - np.exp(-fccd / (beta + sys.float_info.epsilon))) if beta > 0 else 0.0
    if cond_check > alpha:
        msg = " Unphysical parameters: beta * (1-exp(-FCCD/beta)) must be <= alpha. This condition is needed to ensure the smooth transition from exp to linear fccd model transition."
        raise ValueError(msg)

    # Constructing an excellent model.
    if fccd == 0 or alpha == 0:
        trans_pt = 0.0
        exp_norm = 0.0

    elif (beta == 0) or (fccd / beta > math.log(sys.float_info.max)):
        trans_pt = fccd - alpha
        exp_norm = 0.0
    else:

        def f(trans_pt: float) -> float:
            return alpha + trans_pt - fccd + beta * np.exp(-trans_pt / beta) - beta

        # using brentq solver. #Matching this solver with the one used by the Majorana
        # The transition point is between 0 and FCCD, but it must be greater than fccd - alpha to ensure the continuity and differentiability of the function. This is because the linear part starts at fccd - alpha, so the transition point must be greater than this value to ensure a smooth transition between the exponential and linear parts.
        trans_pt = brentq(f, max(0.0, fccd - alpha), fccd)

        # Compute normalization factor for the exponential part of the function
        exp_norm = (beta / alpha) * np.exp(-trans_pt / beta)

    results = np.full_like(distances_flat, np.nan, dtype=np.float64)

    mask_full = (distances_flat >= fccd) | np.isnan(
        distances_flat
    )  # Assigning np.nan with activeness = 1, matching the reboost convention.
    mask_alpha_zero = (
        (distances_flat >= trans_pt) & (alpha == 0.0) & (~mask_full)
    )  # If alpha is zero, the function becomes a step function, so any distance greater than or equal to the transition point is fully active, and any distance less than the transition point is inactive.
    mask_lin = (distances_flat >= trans_pt) & (alpha != 0) & (~mask_full)
    mask_exp_zero = (distances_flat < trans_pt) & (beta == 0.0)
    mask_exp = ~(
        mask_full | mask_alpha_zero | mask_lin | mask_exp_zero
    )  # Safe, avoids recomputing anything expensive

    results[mask_full] = 1.0
    results[mask_alpha_zero] = 0.0
    results[mask_exp_zero] = 0.0
    results[mask_lin] = 1.0 + (distances_flat[mask_lin] - fccd) / (
        alpha + sys.float_info.epsilon
    )  # Adding epsilon to avoid numerical issues when alpha = 0.
    results[mask_exp] = exp_norm * (
        -1.0 + np.exp(distances_flat[mask_exp] / (beta + sys.float_info.epsilon))
    )  # Adding epsilon to avoid numerical issues when beta = 0.

    results = ak.unflatten(ak.Array(results), lengths) if distances_ak.ndim > 1 else results

    return ak.Array(results)
