from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

import h5py
import pygama.evt

from .utils import get_remage_detector_uids

log = logging.getLogger(__name__)


def build_remage_tcm(
    hit_files: str | Path | Iterable[str | Path],
    out_file: str | Path,
    *,
    lh5_group: str = "hit",
    coin_window_in_ns: float = 10_000,
    wo_mode: str = "write_safe",
) -> None:
    """Build the time-coincidence map (TCM) of hit tier files derived from remage output.

    Hits from different detectors belong to the same event when they share the
    Geant4 event identifier ``evtid`` and, sorted in time, their ``t0`` is
    closer than `coin_window_in_ns` to the previous hit. These are the settings
    remage uses, so the TCM is the same remage writes for the stp tier. The
    detector tables are looked up through the ``__by_uid__`` links of
    `lh5_group` (see :func:`reboost.io.write_hit_table_chunk`), so that the
    ``table_key`` field of the TCM holds the remage detector unique identifier.

    Parameters
    ----------
    hit_files
        the hit tier files. The event identifiers must be unique across the
        files, and the ``t0`` fields must be in nanoseconds.
    out_file
        file where the ``tcm`` table is written. Can be one of the hit files.
    lh5_group
        the tier group holding the detector tables.
    coin_window_in_ns
        maximum time difference between consecutive hits of the same event, in
        nanoseconds. The default is the remage default.
    wo_mode
        write mode, forwarded to
        :func:`~pygama.evt.build_tcm.build_tcm`. With the
        default, an existing ``tcm`` table in `out_file` is an error.
    """
    if isinstance(hit_files, str | Path):
        hit_files = [hit_files]

    hit_files = [str(f) for f in hit_files]

    # the coincidence window is in nanoseconds
    for file in hit_files:
        # no file locking: some filesystems do not support it (e.g. CFS at NERSC)
        with h5py.File(file, "r", locking=False) as f:
            for name in get_remage_detector_uids(file, lh5_table=lh5_group).values():
                units = f[f"{lh5_group}/{name}/t0"].attrs.get("units")
                if units is None:
                    log.warning(
                        "%s:%s/%s/t0 has no units, assuming nanoseconds", file, lh5_group, name
                    )
                elif units != "ns":
                    msg = f"{file}:{lh5_group}/{name}/t0 must be in nanoseconds, found '{units}'"
                    raise ValueError(msg)

    pygama.evt.build_tcm(
        [(f, rf"{lh5_group}/__by_uid__/*") for f in hit_files],
        ["evtid", "t0"],
        hash_func=rf"(?<={lh5_group}/__by_uid__/det)\d+",
        coin_windows=[0, coin_window_in_ns],
        out_file=str(out_file),
        wo_mode=wo_mode,
    )
