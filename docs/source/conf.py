# Configuration file for the Sphinx documentation builder.
from __future__ import annotations

import sys
from pathlib import Path

from pkg_resources import get_distribution

sys.path.insert(0, Path(__file__).parents[2].resolve().as_posix())

project = "reboost"
copyright = "The LEGEND Collaboration"
version = get_distribution("reboost").version

extensions = [
    "sphinx.ext.githubpages",
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
    "myst_parser",
]


source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
master_doc = "index"
# language = "python"

# Furo theme
html_theme = "furo"
html_theme_options = {
    "source_repository": "https://github.com/legend-exp/reboost",
    "source_branch": "main",
    "source_directory": "docs/source",
}
html_title = f"{project} {version}"

autodoc_default_options = {"ignore-module-all": True}

# sphinx-napoleon
# enforce consistent usage of NumPy-style docstrings
napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_use_ivar = True

# intersphinx
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("http://docs.scipy.org/doc/numpy", None),
    "scipy": ("http://docs.scipy.org/doc/scipy/reference", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
    "matplotlib": ("http://matplotlib.org/stable", None),
    "pygama": ("https://pygama.readthedocs.io/en/stable/", None),
    "legendhpges": ("https://legend-pygeom-hpges.readthedocs.io/en/latest/", None),
}  # add new intersphinx mappings here


suppress_warnings = [
    # "histogram" is defined both in pygama.dsp.processors.histogram.histogram
    # and in pygama.math.histogram, leading to a Sphinx cross-referencing
    # warning.  I don't know how to fix this properly
    "ref.python",
]
# sphinx-autodoc
# Include __init__() docstring in class docstring
autoclass_content = "both"
autodoc_typehints = "both"
autodoc_typehints_description_target = "documented_params"
autodoc_typehints_format = "short"
