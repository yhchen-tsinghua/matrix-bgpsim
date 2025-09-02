# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

project = 'matrix-bgpsim'
copyright = '2025, Yihao Chen'
author = 'Yihao Chen'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',       # For auto API docs
    'sphinx.ext.napoleon',      # For Google/NumPy style docstrings
    'sphinx.ext.viewcode',      # Add links to source code
    'sphinx_autodoc_typehints', # For type hints
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# The theme to use for HTML pages.
# See https://www.sphinx-doc.org/en/master/usage/theming.html
# html_theme = 'alabaster'
# html_theme = "sphinx_rtd_theme"  # or "furo", "pydata_sphinx_theme", etc.
html_theme = "furo"
# html_theme = "pydata_sphinx_theme"

# If using sphinx_rtd_theme, you also need to import it
# import sphinx_rtd_theme

html_static_path = ['_static']
