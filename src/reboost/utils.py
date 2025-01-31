from __future__ import annotations

import json
import logging
from collections import namedtuple
from pathlib import Path

import yaml

log = logging.getLogger(__name__)


def load_dict(fname: str, ftype: str | None = None) -> dict:
    """Load a text file as a Python dict."""
    __file_extensions__ = {"json": [".json"], "yaml": [".yaml", ".yml"]}

    fname = Path(fname)

    # determine file type from extension
    if ftype is None:
        for _ftype, exts in __file_extensions__.items():
            if fname.suffix in exts:
                ftype = _ftype

    msg = f"loading {ftype} dict from: {fname}"
    log.debug(msg)

    with fname.open() as f:
        if ftype == "json":
            return json.load(f)
        if ftype == "yaml":
            return yaml.safe_load(f)

        msg = f"unsupported file format {ftype}"
        raise NotImplementedError(msg)


def get_wo_mode(
    file_idx: int,
    group_idx: int,
    in_det_idx: int,
    out_det_idx: int,
    chunk_idx: int,
    max_chunk_idx: int,
) -> str:
    raise NotImplementedError


def dict2tuple(dictionary: dict) -> namedtuple:
    return namedtuple("parameters", dictionary.keys())(**dictionary)
