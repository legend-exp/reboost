from __future__ import annotations

import hdf5plugin
import lh5

from ._version import version as __version__
from .build_hit import build_hit

__all__ = [
    "__version__",
    "build_hit",
]

lh5.io.settings.DEFAULT_HDF5_SETTINGS = {"compression": hdf5plugin.Zstd()}
