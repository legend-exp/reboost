from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

import hdf5plugin
import lh5

from ._version import version as __version__
from .io import (
    get_rows_in_event_range,
    init_hit_table,
    read_hit_field_by_tcm,
    write_hit_table_chunk,
)
from .log_utils import setup_log
from .profile import make_profiler
from .tcm import build_remage_tcm
from .utils import get_remage_detector_uids

if TYPE_CHECKING:
    from types import ModuleType

    from . import daq, hpge, io, math, optmap, pmts, shape, spms, units

__all__ = [
    "__version__",
    "build_remage_tcm",
    "daq",
    "get_remage_detector_uids",
    "get_rows_in_event_range",
    "hpge",
    "init_hit_table",
    "io",
    "make_profiler",
    "math",
    "optmap",
    "pmts",
    "read_hit_field_by_tcm",
    "setup_log",
    "shape",
    "spms",
    "units",
    "write_hit_table_chunk",
]

# subpackages pulling heavy dependencies (pyg4ometry, legendhpges, numba) are only imported
# on first access, to keep `import reboost` and the command line tools fast
_LAZY_SUBMODULES = ("daq", "hpge", "io", "math", "optmap", "pmts", "shape", "spms", "units")


def __getattr__(name: str) -> ModuleType:
    if name in _LAZY_SUBMODULES:
        return importlib.import_module(f".{name}", __name__)

    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


lh5.io.settings.DEFAULT_HDF5_SETTINGS = {"compression": hdf5plugin.Zstd()}
