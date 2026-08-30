# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'elyza'
copyright = '2026, Atticus Rex'
author = 'Atticus Rex'
release = '0.1.0'

# pointing sphynx 
import os
import sys
sys.path.insert(0, os.path.abspath('../../src'))

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',   # if you use Google or NumPy style docstrings
    'sphinx.ext.viewcode',   # adds links to highlighted source code
    'myst_nb',               # Markdown support + executes code-cell blocks and embeds output
]

templates_path = ['_templates']
exclude_patterns = []

# -- MyST-NB (code execution) -------------------------------------------------
# https://myst-nb.readthedocs.io/en/latest/configuration.html
nb_execution_mode = 'cache'         # execute notebooks/code-cells and cache results
nb_execution_timeout = 120
nb_execution_raise_on_error = True  # fail the build if a doc's code errors out



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_book_theme'
html_static_path = ['_static']
html_logo = '_static/elyza_logo.png'
html_context = {
    'default_mode': 'dark',
}
html_css_files = ['custom.css']
html_theme_options = {
    "show_nav_level": 1, 
    "navigation_depth": 3,
    "collapse_navigation": True,
}