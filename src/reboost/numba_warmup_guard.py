"""Package-agnostic guard that a numba cache-warmup routine stays complete.

Drop this single file into any package (its source tree or its test dir) and use
it to make sure a *warmup* routine keeps compiling every ``@njit(cache=True)``
function the package defines.

Background
----------
Numba ``@njit(cache=True)`` functions compile lazily, on the first *call* with
concrete argument types: importing a module never populates the on-disk cache.
Packages therefore ship a *warmup* routine that calls every jitted function once
with cheap dummy inputs, so the cache can be pre-built (e.g. baked into a
container image). The risk is that someone adds a new jitted function and forgets
to warm it. This module discovers every numba dispatcher in a package, runs the
package's warmup routine, and asserts each dispatcher actually got compiled.

Usage
-----
Write a warmup routine in your package, e.g. ``mypkg/warmup.py``::

    def warmup() -> None:
        my_kernel(np.zeros(4))
        other_kernel(np.arange(10))

    if __name__ == "__main__":
        warmup()

Then, in a test module, either write the test explicitly::

    from numba_warmup_guard import assert_warmup_covers_all_jitted

    def test_warmup_covers_all_jitted_functions(tmp_path):
        assert_warmup_covers_all_jitted("mypkg", "mypkg.warmup:warmup", tmp_path)

or let the factory build it for you (pytest collects the module-level name)::

    from numba_warmup_guard import make_warmup_test

    test_warmup = make_warmup_test("mypkg", "mypkg.warmup:warmup")

Contract
--------
- Jitted functions are reachable as *module-level attributes* of some submodule
  of the package (the ordinary ``@njit def foo(...)`` at top level). Dispatchers
  hidden in closures or stashed only inside containers are not discovered.
- The warmup entrypoint is an importable, zero-argument callable, named as a
  ``"module:callable"`` string. Whatever it does internally is up to you.

The check runs in a subprocess with a fresh numba cache directory, on purpose: a
test suite's ``conftest`` may force ``cache=False`` (to skip caching / enable
bounds checks), under which inner functions get folded into their callers
without registering their own signatures. Only a clean interpreter reproduces
the real production behaviour (``cache=True``, one cache file per function).

This module imports nothing beyond the standard library; numba is imported only
inside the subprocess it spawns.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

# Fully generic: argv[1] is the package to scan, argv[2] a "module:callable"
# warmup entrypoint. Discover every numba dispatcher in the package, run the
# consumer's warmup callable, then report how many dispatchers exist and which
# were left uncompiled.
_CHILD_SCRIPT = textwrap.dedent(
    """
    import importlib, pkgutil, sys
    from numba.core.registry import CPUDispatcher

    pkg_name = sys.argv[1]
    warm_mod_name, _, warm_func_name = sys.argv[2].partition(":")
    pkg = importlib.import_module(pkg_name)

    disp = {}
    for m in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + "."):
        if m.name == warm_mod_name:
            continue
        mod = importlib.import_module(m.name)
        for o in vars(mod).values():
            if isinstance(o, CPUDispatcher) and o.py_func.__module__.startswith(pkg_name):
                disp[f"{o.py_func.__module__}.{o.py_func.__qualname__}"] = o

    # run the consumer-provided warmup routine, however it chooses to warm
    warm_mod = importlib.import_module(warm_mod_name)
    getattr(warm_mod, warm_func_name)()

    missing = sorted(k for k, d in disp.items() if not d.signatures)
    print("WARMUP_TOTAL:", len(disp))
    print("WARMUP_MISSING:", ",".join(missing))
    """
)


def _run_child(package: str, warmup_entrypoint: str, tmp_path) -> dict[str, str]:
    env = {**os.environ, "NUMBA_CACHE_DIR": str(tmp_path / "numba-cache")}

    proc = subprocess.run(
        [sys.executable, "-c", _CHILD_SCRIPT, package, warmup_entrypoint],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert proc.returncode == 0, f"warmup subprocess crashed:\n{proc.stdout}\n{proc.stderr}"

    # a warmup routine may emit unrelated text to stdout, so parse only our
    # sentinel lines.
    out = {}
    for line in proc.stdout.splitlines():
        for tag in ("WARMUP_TOTAL", "WARMUP_MISSING"):
            if line.startswith(tag + ":"):
                out[tag] = line.split(":", 1)[1].strip()
    return out


def assert_warmup_covers_all_jitted(package: str, warmup_entrypoint: str, tmp_path) -> None:
    """Assert running ``warmup_entrypoint`` compiles every jitted function in ``package``.

    Parameters
    ----------
    package
        Import name of the package to scan for numba dispatchers, e.g. ``"mypkg"``.
    warmup_entrypoint
        A ``"module:callable"`` string naming a zero-argument callable that warms
        the cache, e.g. ``"mypkg.warmup:warmup"``.
    tmp_path
        A pytest ``tmp_path`` (used as a throwaway numba cache directory).
    """
    out = _run_child(package, warmup_entrypoint, tmp_path)

    assert int(out["WARMUP_TOTAL"]) > 0, f"no numba dispatchers discovered in {package}"

    missing = [k for k in out["WARMUP_MISSING"].split(",") if k]
    assert not missing, (
        f"these numba-jitted functions were not warmed up by {warmup_entrypoint} "
        f"(add a call for each): {missing}"
    )


def make_warmup_test(package: str, warmup_entrypoint: str):
    """Build a ``test_*`` function that pytest collects; see :func:`assert_warmup_covers_all_jitted`."""

    def test_warmup_covers_all_jitted_functions(tmp_path):
        assert_warmup_covers_all_jitted(package, warmup_entrypoint, tmp_path)

    return test_warmup_covers_all_jitted_functions
