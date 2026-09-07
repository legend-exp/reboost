from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from pathlib import Path

import h5py
from lgdo.types import VectorOfVectors

log = logging.getLogger(__name__)


def get_table_names(tcm: VectorOfVectors) -> dict:
    """Extract table names from tcm.attrs['tables'] and return them as a dictionary."""
    raw = tcm.attrs["tables"]
    cleaned = raw.strip("[]").replace(" ", "").replace("'", "")
    tables = cleaned.split(",")
    tables = [tab.split("/")[-1] for tab in tables]

    return {name: idx for idx, name in enumerate(tables)}


def _check_input_file(parser, file: str | Iterable[str], descr: str = "input") -> None:
    file = (file,) if isinstance(file, str) else file
    not_existing = [f for f in file if not Path(f).exists()]
    if not_existing != []:
        parser.error(f"{descr} file(s) {''.join(not_existing)} missing")


def _check_output_file(parser, file: str | Iterable[str] | None, optional: bool = False) -> None:
    if file is None and optional:
        return

    assert file is not None
    files_list: Iterable[str] = (file,) if isinstance(file, str) else file
    for f in files_list:
        if Path(f).exists():
            parser.error(f"output file {f} already exists")


def get_remage_detector_uids(h5file: str | Path, *, lh5_table: str = "stp") -> dict:
    """Get mapping of detector names to UIDs from a remage output file.

    The remage LH5 output files contain a link structure that lets the user
    access detector tables by UID. For example:

    .. code-block:: text

        ├── stp · struct{det1,det2,optdet1,optdet2,scint1,scint2}
        └── __by_uid__ · struct{det001,det002,det011,det012,det101,det102}
            ├── det001 -> /stp/scint1
            ├── det002 -> /stp/scint2
            ├── det011 -> /stp/det1
            ├── det012 -> /stp/det2
            ├── det101 -> /stp/optdet1
            └── det102 -> /stp/optdet2

    This function analyzes this structure and returns:

    .. code-block:: text

        {1: 'scint1',
         2: 'scint2',
         11: 'det1',
         12: 'det2',
         101: 'optdet1',
         102: 'optdet2'g

    Parameters
    ----------
    h5file
        path to remage output file.
    """
    if isinstance(h5file, Path):
        h5file = h5file.as_posix()

    out = {}
    with h5py.File(h5file, "r") as f:
        g = f[f"/{lh5_table}/__by_uid__"]
        # loop over links
        for key in g:
            # is this a link?
            link = g.get(key, getlink=True)
            if isinstance(link, h5py.SoftLink):
                m = re.fullmatch(r"det(\d+)", key)
                if m is None:
                    msg = rf"'{key}' is not formatted as expected, i.e. 'det(\d+)', skipping"
                    log.warning(msg)
                    continue

                # get the name of the link target without trailing groups (to
                # i.e. remove /stp)
                name = link.path.split("/")[-1]

                out[int(m.group(1))] = name
    return out
