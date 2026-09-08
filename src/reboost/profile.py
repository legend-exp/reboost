from __future__ import annotations

import contextlib
import logging
import time
from collections import defaultdict
from collections.abc import Callable, Generator

import psutil

log = logging.getLogger(__name__)


def _f(x: float) -> str:
    return f"{x:.5g}"


def _pct(x: float) -> str:
    return f"{x:.1f}"


def make_profiler() -> tuple[Callable, Callable, Callable]:
    """Create a profiler for the wall time and memory footprint of blocks of code.

    Returns three functions. The first is a context manager that measures the
    code in the ``with`` statement and adds it to the block with the given
    name. The second logs, for each block, the total wall time and the change
    of resident memory of the process (maximum and average over the calls, in
    megabytes). The third logs the wall time spent in each block since its last
    call. The reports are logged at the ``INFO`` level of the
    ``reboost.profile`` logger, single calls at the ``DEBUG`` level.

    Examples
    --------
    >>> profile_block, print_stats, print_stats_since_last = make_profiler()
    >>> for chunk in iterator:
    ...     with profile_block("distance_to_surface()"):
    ...         dist = distance_to_surface(...)
    >>> print_stats()
    """
    proc = psutil.Process()
    stats: dict[str, dict[str, float]] = defaultdict(
        lambda: {
            "wall_s": 0.0,
            # per-call rss deltas (rss_after - rss_before);
            # we keep both the maximum delta and the average delta
            "max_delta_rss_mb": float("-inf"),
            "avg_delta_rss_mb": 0.0,
            "n_calls": 0,
        }
    )
    last_stats: dict[str, dict[str, float]] = {}

    def _print_report(title: str, view_stats: dict) -> None:
        log.info(title)

        blocks = list(view_stats)
        sum_wall_s = sum(view_stats[b]["wall_s"] for b in blocks)
        sum_max_delta_rss_mb = sum(view_stats[b]["max_delta_rss_mb"] for b in blocks)
        sum_avg_delta_rss_mb = sum(view_stats[b]["avg_delta_rss_mb"] for b in blocks)

        for block in blocks:
            s = view_stats[block]

            wall_s = s["wall_s"]
            max_delta_rss_mb = s["max_delta_rss_mb"]
            avg_delta_rss_mb = s["avg_delta_rss_mb"]

            wall_s_frac = 100.0 * wall_s / sum_wall_s if sum_wall_s else 0.0
            max_delta_rss_mb_frac = (
                100.0 * max_delta_rss_mb / sum_max_delta_rss_mb if sum_max_delta_rss_mb else 0.0
            )
            avg_delta_rss_mb_frac = (
                100.0 * avg_delta_rss_mb / sum_avg_delta_rss_mb if sum_avg_delta_rss_mb else 0.0
            )

            msg = (
                f"block {block} ]]] "
                f"wall_time_s={_f(wall_s)} ({_pct(wall_s_frac)}%) "
                f"max_delta_rss_mb={_f(max_delta_rss_mb)} ({_pct(max_delta_rss_mb_frac)}%) "
                f"avg_delta_rss_mb={_f(avg_delta_rss_mb)} ({_pct(avg_delta_rss_mb_frac)}%)"
            )
            log.info(msg)

        msg = "=========================="
        log.info(msg)

    @contextlib.contextmanager
    def profile_block(name: str) -> Generator[None, None, None]:
        t0 = time.perf_counter()
        rss0 = proc.memory_info().rss

        try:
            yield
        finally:
            t1 = time.perf_counter()
            rss1 = proc.memory_info().rss

            delta_mb = (rss1 - rss0) / 1024**2

            s = stats[name]
            s["wall_s"] += t1 - t0

            # update max delta
            s["max_delta_rss_mb"] = max(s["max_delta_rss_mb"], delta_mb)

            # online mean update for the average delta
            s["n_calls"] += 1
            s["avg_delta_rss_mb"] += (delta_mb - s["avg_delta_rss_mb"]) / s["n_calls"]

            msg = f"block {name}: delta_rss_mb={_f(delta_mb)} wall_time_s={_f(t1 - t0)} "
            log.debug(msg)

    def print_stats() -> None:
        _print_report("==== profiling report ====", stats)

    def print_stats_since_last() -> None:
        nonlocal last_stats

        view = {}
        for block, cur in stats.items():
            prev = last_stats.get(
                block,
                {
                    "wall_s": 0.0,
                    "max_delta_rss_mb": float("-inf"),
                    "avg_delta_rss_mb": 0.0,
                },
            )

            view[block] = {
                "wall_s": cur["wall_s"] - prev["wall_s"],
                "max_delta_rss_mb": cur["max_delta_rss_mb"],
                "avg_delta_rss_mb": cur["avg_delta_rss_mb"],
            }

        _print_report("==== profiling report (since last) ====", view)
        last_stats = {block: dict(stats[block]) for block in stats}

    return profile_block, print_stats, print_stats_since_last
