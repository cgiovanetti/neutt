"""Sphinx configuration for the neutt API reference."""

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

import neutt  # noqa: E402  (enables float64, as every user import does)

project = "neutt"
author = "Cara Giovanetti, Evan Grohs, George Fuller, Shasha Arani"
copyright = "2026, " + author
release = neutt.__version__
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "myst_parser",
]

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
exclude_patterns = ["_build"]

autosummary_generate = True
autodoc_default_options = {"members": True, "member-order": "bysource"}
autodoc_typehints = "description"
napoleon_google_docstring = False
napoleon_numpy_docstring = True

myst_enable_extensions = ["dollarmath"]
myst_heading_anchors = 2

html_theme = "furo"
html_title = f"neutt {release}"
