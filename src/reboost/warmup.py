"""Warm up the on-disk numba cache.

Numba ``@njit(cache=True)`` functions compile lazily, on the first *call*:
importing a module never populates the cache. :func:`warmup` calls every jitted
function in :mod:`reboost` once, with small dummy inputs, so the cache is
written ahead of first real use (e.g. in a freshly built container)::

    python -m reboost.warmup

``tests/test_warmup.py`` fails if a new jitted function is added without a call
here.
"""

from __future__ import annotations

import awkward as ak
import numpy as np

from . import units
from .daq import run_daq_non_sparse
from .hpge import psd, surface
from .optmap import convolve
from .optmap.create import _compute_hit_maps
from .shape import cluster
from .spms.pe import (
    detected_photoelectrons,
    emitted_scintillation_photons,
    number_of_detected_photoelectrons,
    photoelectron_times,
)


def warmup() -> None:
    """Call every cached numba-jitted function once, to populate the on-disk cache."""
    # daq
    run_daq_non_sparse(
        ak.Array(
            {
                "evtid": [0, 1, 2],
                "geds_energy_active": [[100.0, 50.0], [30.0], [200.0, 150.0]],
                "geds_rawid_active": [[1000, 2000], [1000], [1000, 2000]],
            }
        ),
        n_sim_events=100,
        source_activity=1.0,
    )

    # hpge.psd + hpge.surface
    model, x = psd.get_current_template(
        -100, 100, 1.0, amax=1, mean_aoe=1, mu=0, sigma=10, tau=10,
        tail_fraction=0.65, high_tail_fraction=0.1, high_tau=10,
    )  # fmt: skip
    edep = units.attach_units(ak.Array([[100.0, 50.0], [10.0]]), "keV")
    drift_time = units.attach_units(ak.Array([[10.0, 20.0], [15.0]]), "ns")
    psd.maximum_current(edep, drift_time, template=model, times=x)
    psd.drift_time_heuristic(drift_time, edep)
    psd.get_current_waveform(
        np.array([100.0, 50.0]), np.array([10.0, 20.0]), model,
        float(x[0]), 1.0, (float(x[0]), float(x[-1])),
    )  # fmt: skip
    charge = np.zeros(20)
    charge[10] = 1.0
    surface._compute_diffusion_impl(charge, nsteps=10, factor=0.29)

    # shape.cluster (dist_to_surf=None and =<array> are distinct specializations)
    trackid = ak.Array([[0, 0, 0], [1, 1]])
    xloc = ak.Array([[0.0, 0.1, 5.0], [0.0, 0.1]])
    yloc = ak.full_like(xloc, 0.0)
    zloc = ak.full_like(xloc, 0.0)
    dist = ak.Array([[1.0, 0.05, 2.0], [1.0, 0.05]])
    cluster.cluster_by_step_length(trackid, xloc, yloc, zloc)
    cluster.cluster_by_step_length(
        trackid, xloc, yloc, zloc, dist, threshold_in_mm=1, threshold_surf_in_mm=1, surf_cut=0.1
    )

    # optmap + spms.pe
    edges = np.linspace(0.0, 1.0, 5)
    optmap = convolve.OptmapForConvolve(
        np.array(["all"]), np.array([0]), (edges, edges, edges),
        np.full((1, 4, 4, 4), 0.1, dtype=np.float64),
    )  # fmt: skip
    edep = ak.Array([[1.0, 2.0], [3.0]])
    particle = ak.Array([[22, 22], [22]])
    nph = ak.Array([[10, 20], [30]])
    xloc = ak.Array([[0.1, 0.2], [0.3]])
    yloc = ak.Array([[0.1, 0.2], [0.3]])
    zloc = ak.Array([[0.1, 0.2], [0.3]])
    time = ak.Array([[0.0, 1.0], [2.0]])
    num_det_ph = emitted_scintillation_photons(edep, particle, "lar")
    number_of_detected_photoelectrons(xloc, yloc, zloc, nph, optmap, "all")
    number_of_detected_photoelectrons(xloc, yloc, zloc, nph, optmap, "all", max_pes_per_hit=5)
    photoelectron_times(num_det_ph, particle, time, "lar")
    detected_photoelectrons(nph, particle, time, xloc, yloc, zloc, optmap, "lar", "all")
    hitcounts = np.zeros((5, 2), dtype=np.int64)
    hitcounts[0, 0] = 1
    hitcounts[2, 1] = 1
    _compute_hit_maps(hitcounts, 3, np.array([0, 1], dtype=np.int64))


if __name__ == "__main__":
    warmup()
