from __future__ import annotations

from collections.abc import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import ArrayLike, NDArray

from .utils import HPGePulseShapeLibrary, HPGeRZField


def symmetrize(field: ArrayLike) -> NDArray:
    """Mirror an `(r, z)` map about `r = 0`, for display as a detector section."""
    field = np.asarray(field).T
    return np.concatenate((np.fliplr(field), field), axis=1)


def _psl_bulk_aoe(library: HPGePulseShapeLibrary) -> float:
    """Most common A/E over the library, i.e. the value of the detector bulk."""
    waveforms = np.asarray(library.waveforms)
    inside = ~np.isnan(waveforms).all(axis=-1)

    peaks = np.nanmax(waveforms[inside], axis=-1)
    counts, edges = np.histogram(peaks, bins=1000)

    return float(((edges[:-1] + edges[1:]) / 2)[counts.argmax()])


def _psl_aoe_map(library: HPGePulseShapeLibrary, *, normalise: bool = True) -> NDArray:
    """A/E of every grid point, the maximum of its current pulse, NaN outside the detector."""
    waveforms = np.asarray(library.waveforms)

    # a grid point outside the detector holds no pulse
    inside = ~np.isnan(waveforms).all(axis=-1)

    aoe = np.full(waveforms.shape[:2], np.nan)
    aoe[inside] = np.nanmax(waveforms[inside], axis=-1)

    if normalise:
        aoe /= _psl_bulk_aoe(library)

    return aoe


def plot_rz_maps(
    maps: Mapping[int, ArrayLike],
    r: ArrayLike,
    z: ArrayLike,
    *,
    hpge: object | None = None,
    label: str = "",
    title: str = "",
    vmin: float | None = None,
    vmax: float | None = None,
    axes: Sequence[Axes] | None = None,
) -> tuple[Figure, NDArray]:
    r"""Draw one or more `(r, z)` maps of a detector, side by side.

    Each map is mirrored about `r = 0` and drawn as a section of the detector.
    The maps share a colour scale. When the maps of both crystal axes are given,
    their ratio is drawn in an extra panel.

    Pixels holding NaN, outside the detector, are left blank.

    Parameters
    ----------
    maps
        the maps to draw, keyed by the azimuth of the crystal axis in degrees.
        Each is a 2D array over the `(r, z)` grid.
    r
        radial coordinates of the grid, in mm.
    z
        axial coordinates of the grid, in mm, with the p+ contact at `z = 0`.
    hpge
        detector object, as returned by ``pygeomhpges.make_hpge``. Its profile
        is drawn on every panel.
    label
        label of the colour bar, e.g. ``"drift time [ns]"``.
    title
        prefixed to the title of every panel, e.g. the detector name.
    vmin, vmax
        limits of the colour scale. Taken from the maps if not given.
    axes
        draw into these axes, one per panel, instead of making a figure. Use it
        to choose the size of the figure or to place the panels among others::

            fig, axes = plt.subplots(ncols=3, figsize=(14, 6), sharey=True)
            plot_rz_maps(maps, r, z, axes=axes)

    Returns
    -------
    the figure and its axes.
    """
    angles = sorted(maps)
    if not angles:
        msg = "no maps to draw"
        raise ValueError(msg)

    # crystal axes, by azimuth in degrees
    labels = {0: r"$\langle 100 \rangle$", 45: r"$\langle 110 \rangle$"}

    images = {angle: symmetrize(maps[angle]) for angle in angles}
    r, z = np.asarray(r), np.asarray(z)
    extent = (-r.max(), r.max(), z.min(), z.max())

    with_ratio = {0, 45}.issubset(angles)
    n_panels = len(angles) + (1 if with_ratio else 0)

    if vmin is None:
        vmin = float(np.nanmin(list(images.values())))
    if vmax is None:
        vmax = float(np.nanmax(list(images.values())))

    if axes is None:
        fig, axes = plt.subplots(
            ncols=n_panels, figsize=(4 * n_panels, 4), sharey=True, squeeze=False
        )
        axes = axes[0]
    else:
        axes = np.atleast_1d(np.asarray(axes, dtype=object))
        if len(axes) != n_panels:
            msg = f"{n_panels} panels to draw, but {len(axes)} axes were given"
            raise ValueError(msg)
        fig = axes[0].get_figure()

    def draw(ax, image, panel_title, cmap="viridis", **kwargs):
        im = ax.imshow(image, origin="lower", extent=extent, aspect="equal", cmap=cmap, **kwargs)
        if hpge is not None:
            from pygeomhpges.draw import plot_profile  # noqa: PLC0415

            plot_profile(hpge, axes=ax, marker=None, linewidth=1, color="black")

        # leave a margin around the detector
        margin = 0.04
        ax.set_xlim(
            extent[0] - margin * (extent[1] - extent[0]),
            extent[1] + margin * (extent[1] - extent[0]),
        )
        ax.set_ylim(
            extent[2] - margin * (extent[3] - extent[2]),
            extent[3] + margin * (extent[3] - extent[2]),
        )
        ax.set_xlabel("r [mm]")
        ax.set_title(f"{title} {panel_title}".strip())
        return im

    im = None
    for ax, angle in zip(axes[: len(angles)], angles, strict=True):
        im = draw(ax, images[angle], labels.get(angle, f"{angle}°"), vmin=vmin, vmax=vmax)

    axes[0].set_ylabel("z [mm]")
    fig.colorbar(im, ax=axes[: len(angles)], label=label)

    if with_ratio:
        with np.errstate(divide="ignore", invalid="ignore"):
            values = np.divide(
                images[0], images[45], out=np.full_like(images[0], np.nan), where=images[45] > 0
            )
        spread = np.nanmax(np.abs(values - 1)) or 1.0
        im = draw(
            axes[-1],
            values,
            f"{labels[0]} / {labels[45]}",
            cmap="coolwarm",
            vmin=1 - spread,
            vmax=1 + spread,
        )
        fig.colorbar(im, ax=axes[-1], label="ratio")

    return fig, axes


def _grid_of(field: HPGeRZField) -> tuple[NDArray, NDArray, NDArray]:
    """The grid of a field and its values, with the coordinates in mm."""
    from .. import units  # noqa: PLC0415

    if field.values is None:
        msg = "the field carries no gridded values, it cannot be drawn"
        raise ValueError(msg)

    r = (field.r * units.ureg.Quantity(1, field.r_units)).to("mm").m
    z = (field.z * units.ureg.Quantity(1, field.z_units)).to("mm").m
    return r, z, field.values


def plot_drift_time_maps(
    dt_maps: Mapping[int, HPGeRZField], *, hpge: object | None = None, **kwargs
) -> tuple[Figure, NDArray]:
    """Draw the drift-time maps of a detector, one panel per crystal axis.

    Parameters
    ----------
    dt_maps
        drift-time maps keyed by the azimuth of the crystal axis in degrees, as
        returned by :func:`reboost.hpge.load_hpge_drift_time_maps`.
    hpge
        detector object, whose profile is drawn on every panel.
    **kwargs
        forwarded to :func:`plot_rz_maps`.

    Returns
    -------
    the figure and its axes.
    """
    grids = {angle: _grid_of(field) for angle, field in dt_maps.items()}
    r, z, _ = next(iter(grids.values()))
    maps = {angle: values for angle, (_, _, values) in grids.items()}

    kwargs.setdefault("label", "drift time [ns]")
    return plot_rz_maps(maps, r, z, hpge=hpge, **kwargs)


def plot_psl_aoe_maps(
    libraries: Mapping[int, HPGePulseShapeLibrary],
    *,
    hpge: object | None = None,
    normalise: bool = True,
    **kwargs,
) -> tuple[Figure, NDArray]:
    """Draw the A/E of a detector over the `(r, z)` plane, one panel per crystal axis.

    A/E of a grid point is the maximum of its current pulse, i.e. the value a
    single energy deposition there would give.

    Parameters
    ----------
    libraries
        pulse-shape libraries of *current* waveforms, keyed by the azimuth of
        the crystal axis in degrees, as returned by
        :func:`reboost.hpge.load_hpge_pulse_shape_libraries`.
    hpge
        detector object, whose profile is drawn on every panel.
    normalise
        divide by the most common A/E, the value of the detector bulk, so that
        the bulk sits at one.
    **kwargs
        forwarded to :func:`plot_rz_maps`.

    Returns
    -------
    the figure and its axes.
    """
    maps = {angle: _psl_aoe_map(lib, normalise=normalise) for angle, lib in libraries.items()}
    first = next(iter(libraries.values()))

    from .. import units  # noqa: PLC0415

    r = (first.r * units.ureg.Quantity(1, first.r_units)).to("mm").m
    z = (first.z * units.ureg.Quantity(1, first.z_units)).to("mm").m

    kwargs.setdefault("label", "A/E")
    return plot_rz_maps(maps, r, z, hpge=hpge, **kwargs)
