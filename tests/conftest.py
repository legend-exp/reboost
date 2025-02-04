from __future__ import annotations

import shutil
import uuid
from getpass import getuser
from pathlib import Path
from tempfile import gettempdir

import numba
import pytest

_tmptestdir = Path(gettempdir()) / Path(f"reboost-ests-{getuser()}-{uuid.uuid4()!s}")


@pytest.fixture(scope="session")
def tmptestdir():
    Path.mkdir(_tmptestdir)
    return _tmptestdir


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


patch_numba_for_tests()
