from __future__ import annotations

import logging
import time

from reboost.profile import make_profiler


def test_make_profiler_returns_three_callables():
    profile_block, print_stats, print_stats_since_last = make_profiler()

    assert callable(profile_block)
    assert callable(print_stats)
    assert callable(print_stats_since_last)


def test_profiler_context_manager_and_printers_are_callable():
    profile_block, print_stats, print_stats_since_last = make_profiler()

    with profile_block("block_a"):
        pass

    with profile_block("block_b"):
        pass

    print_stats()
    print_stats_since_last()
    print_stats_since_last()


def test_profiler_reports(caplog):
    profile_block, print_stats, print_stats_since_last = make_profiler()

    with profile_block("block_a"):
        time.sleep(0.01)

    with profile_block("block_a"):
        time.sleep(0.01)

    with profile_block("block_b"):
        pass

    with caplog.at_level(logging.INFO, logger="reboost.profile"):
        print_stats()

    lines = [r.message for r in caplog.records]
    assert lines[0] == "==== profiling report ===="
    assert lines[1].startswith("block block_a ]]] wall_time_s=")
    assert lines[2].startswith("block block_b ]]] wall_time_s=")
    assert "max_delta_rss_mb=" in lines[1]
    assert "avg_delta_rss_mb=" in lines[1]

    wall_s = float(lines[1].split("wall_time_s=")[1].split(" ")[0])
    assert wall_s >= 0.02

    caplog.clear()

    with caplog.at_level(logging.INFO, logger="reboost.profile"):
        print_stats_since_last()

        with profile_block("block_a"):
            time.sleep(0.01)

        print_stats_since_last()
        print_stats()

    lines = [r.message for r in caplog.records]
    assert lines[0] == "==== profiling report (since last) ===="

    # the two "since last" reports split the total time of block_a: the first
    # one covers the two initial calls, the second one only the last call
    block_a = [line for line in lines if line.startswith("block block_a")]
    assert len(block_a) == 3
    wall_s = [float(line.split("wall_time_s=")[1].split(" ")[0]) for line in block_a]
    assert wall_s[0] >= 0.02
    assert wall_s[1] >= 0.01
    assert abs(wall_s[0] + wall_s[1] - wall_s[2]) <= 1e-3 * wall_s[2]
