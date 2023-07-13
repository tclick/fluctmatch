"""Sphinx configuration."""
project = "Fluctuation matching"
author = "Timothy H. Click"
copyright = "2013-2023, Timothy H. Click"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_click",
    "myst_parser",
]
autodoc_typehints = "description"
html_theme = "furo"
