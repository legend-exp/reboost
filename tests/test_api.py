from __future__ import annotations

import importlib

import pytest

import reboost

# subpackage -> modules whose public names it re-exports
SUBPACKAGES = {
    "daq": ["core"],
    "hpge": ["psd", "surface", "utils"],
    "math": ["functions", "stats"],
    "pmts": ["functions"],
    "shape": ["cluster", "group"],
    "spms": ["pe"],
}


@pytest.mark.parametrize("name", sorted(reboost.__all__))
def test_toplevel_names_resolve(name):
    assert getattr(reboost, name) is not None


@pytest.mark.parametrize("subpkg", sorted(SUBPACKAGES))
def test_reexports_are_the_original_objects(subpkg):
    pkg = importlib.import_module(f"reboost.{subpkg}")

    for name in pkg.__all__:
        obj = getattr(pkg, name)

        # the re-export must be the very same object as the one defined in the leaf module
        origins = [
            getattr(importlib.import_module(f"reboost.{subpkg}.{mod}"), name, None)
            for mod in SUBPACKAGES[subpkg]
        ]
        assert obj in origins


@pytest.mark.parametrize("subpkg", sorted(SUBPACKAGES))
def test_all_public_names_are_reexported(subpkg):
    pkg = importlib.import_module(f"reboost.{subpkg}")

    for mod in SUBPACKAGES[subpkg]:
        module = importlib.import_module(f"reboost.{subpkg}.{mod}")
        public = {
            name
            for name, obj in vars(module).items()
            if not name.startswith("_")
            and getattr(obj, "__module__", None) == module.__name__
            and callable(obj)
        }
        assert public <= set(pkg.__all__)


def test_unknown_attribute_raises():
    with pytest.raises(AttributeError):
        _ = reboost.does_not_exist
