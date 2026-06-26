from __future__ import annotations

import importlib.util
import types

from numba import njit


def numba_pdgid_funcs():
    """Load a numby-optimized copy of the scikit-hep/particle package."""

    def _import_module(mod: str):
        spec = importlib.util.find_spec(mod)
        assert spec is not None
        modobj = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(modobj)
        return modobj

    pdg_func = _import_module("particle.pdgid.functions")
    pdg_version = _import_module("particle.version")

    def _digit2(pdgid, loc: int) -> int:
        e = 10 ** (loc - 1)
        return (pdgid // e % 10) if pdgid >= e else 0

    # version from particle < 1
    def charge2(pdgid) -> float | None:
        """Returns the charge."""
        three_charge_pdgid = pdg_func.three_charge(pdgid)
        if three_charge_pdgid is None:
            return None
        if not pdg_func.is_Qball(pdgid):
            return three_charge_pdgid / 3.0
        return three_charge_pdgid / 30.0

    pdg_func._digit = _digit2
    if pdg_version.version_tuple[0] >= 1:
        pdg_func.charge = charge2

    for fname, f in pdg_func.__dict__.items():
        if not callable(f) or not isinstance(f, types.FunctionType):
            continue
        setattr(pdg_func, fname, njit(f, cache=True))

    return pdg_func
