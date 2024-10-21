from __future__ import annotations

from collections.abc import Iterable

from reboost.hpge import processors, utils


def build_hit(
    lh5_in_file: str, lh5_out_file: str, detectors: Iterable[str | int], buffer_len: int = int(5e6)
) -> None:
    for idx, d in enumerate(detectors):
        delete_input = bool(idx == 0)
        utils.read_write_incremental(
            lh5_out_file,
            f"hit/{d}",
            processors.group_by_event,
            f"hit/{d}",
            lh5_in_file,
            buffer_len,
            delete_input=delete_input,
        )
