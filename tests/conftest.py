from __future__ import annotations

import shutil
import uuid
from getpass import getuser
from pathlib import Path
from tempfile import gettempdir

import numba
import numpy as np
import pytest
from legendtestdata import LegendTestData
from lgdo import Array, Scalar, Struct, lh5

from reboost.hpge import psd
from reboost.optmap.convolve import OptmapForConvolve

_tmptestdir = Path(gettempdir()) / f"reboost-tests-{getuser()}-{uuid.uuid4()!s}"


@pytest.fixture(scope="session")
def tmptestdir_global():
    _tmptestdir.mkdir(exist_ok=False)
    return _tmptestdir


@pytest.fixture(scope="session")
def legendtestdata():
    ldata = LegendTestData()
    ldata.checkout("0a56ff3")
    return ldata


@pytest.fixture(scope="module")
def tmptestdir(tmptestdir_global, request):
    p = tmptestdir_global / request.module.__name__
    p.mkdir(exist_ok=True)  # note: will be cleaned up globally.
    return p


def pytest_sessionfinish(exitstatus):
    if exitstatus == 0 and Path.exists(_tmptestdir):
        shutil.rmtree(_tmptestdir)


def patch_numba_for_tests():
    """Globally disable numba cache and enable bounds checking (for testing)."""
    njit_old = numba.njit

    def njit_patched(*args, **kwargs):
        kwargs.update({"cache": False, "boundscheck": True})
        return njit_old(*args, **kwargs)

    numba.njit = njit_patched


@pytest.fixture(scope="module")
def mock_optmap_for_convolve():
    edges_1d = np.linspace(0.0, 1.0, 11)
    edges = (edges_1d, edges_1d, edges_1d)
    weights = np.full((1, 10, 10, 10), 0.1, dtype=np.float64)
    return OptmapForConvolve(np.array(["all"]), np.array([0]), edges, weights)


@pytest.fixture(scope="module")
def test_pulse_shape_library(tmptestdir):
    model, _ = psd.get_current_template(
        -1000,
        3000,
        1.0,
        amax=1,
        mean_aoe=1,
        mu=0,
        sigma=100,
        tau=100,
        tail_fraction=0.65,
        high_tail_fraction=0.1,
        high_tau=10,
    )

    # loop
    r = z = np.linspace(0, 100, 200)
    waveforms = np.zeros((200, 200, 4001))
    for i in range(200):
        for j in range(200):
            waveforms[i, j] = model

    t0 = -1000
    dt = 1

    res = Struct(
        {
            "r": Array(r, attrs={"units": "mm"}),
            "z": Array(z, attrs={"units": "mm"}),
            "waveforms": Array(waveforms, attrs={"units": ""}),
            "dt": Scalar(dt, attrs={"units": "ns"}),
            "t0": Scalar(t0, attrs={"units": "ns"}),
        }
    )
    lh5.write(res, "V01", f"{tmptestdir}/pulse_shape_lib.lh5")

    return f"{tmptestdir}/pulse_shape_lib.lh5"


@pytest.fixture(scope="function")
def compare_numba_vs_python():
    """Compare the JIT and Python (py_func) versions of an ``@njit`` function.

    Inspired by the `dspeed
    <https://dspeed.readthedocs.io/en/stable/developer.html>`_ approach for
    testing numba-wrapped functions. Both the JIT-compiled version and the pure
    Python version (via ``.py_func``) are called with the same arguments. For
    deterministic functions the outputs are asserted to be numerically equal.
    The JIT result is returned so it can be used directly in assertions.
    """

    def _compare(func, *args, check_equal=True, **kwargs):
        result_jit = func(*args, **kwargs)
        result_py = func.py_func(*args, **kwargs)

        if check_equal:
            jit_outs = result_jit if isinstance(result_jit, tuple) else (result_jit,)
            py_outs = result_py if isinstance(result_py, tuple) else (result_py,)
            for o_jit, o_py in zip(jit_outs, py_outs):
                np.testing.assert_allclose(
                    np.asarray(o_jit, dtype=float),
                    np.asarray(o_py, dtype=float),
                    equal_nan=True,
                )

        return result_jit

    return _compare


patch_numba_for_tests()
