# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join('..', 'tunax')))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'tunax'
author = 'Gabriel Mouttapa'
copyright = '2024, Gabriel Mouttapa'
version = '0.1'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    "sphinx.ext.autosummary",
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx_autodoc_typehints',
    'sphinx.ext.mathjax',
    "sphinx.ext.duration",
    "myst_parser",
    'sphinx.ext.intersphinx',
]
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Napoleon configuration --------------------------------------------------
napoleon_google_docstring = False
napoleon_numpy_docstring = True
autodoc_typehints = "description"
intersphinx_mapping = {
    'jax': ('https://jax.readthedocs.io/en/latest/', None),
    'python': ('https://docs.python.org/3', None),
    'xarray': ('https://docs.xarray.dev/en/stable/', None)
}

# -- Options for markdown files ----------------------------------------------
myst_enable_extensions = ["dollarmath", "amsmath"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
