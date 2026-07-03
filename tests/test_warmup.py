"""Guard test: ``reboost.warmup`` must compile every jitted function in reboost.

The reusable harness lives in :mod:`reboost.numba_warmup_guard` (a drop-in,
package-agnostic module); here we only point it at this package and its warmup
routine.
"""

from __future__ import annotations

from reboost.numba_warmup_guard import make_warmup_test

test_warmup_covers_all_jitted_functions = make_warmup_test("reboost", "reboost.warmup:warmup")
