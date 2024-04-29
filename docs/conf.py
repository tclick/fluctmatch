"""Sphinx configuration."""

project = "Fluctuation Matching"
author = "Timothy H. Click, Ph.D."
copyright = "2013-2024, Timothy H. Click, Ph.D."
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_click",
    "myst_parser",
]
autodoc_typehints = "description"
html_theme = "furo"
