# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import sys
from pathlib import Path
import sphinx_book_theme

sys.path.insert(0, str(Path('..', 'src').resolve()))
import helixnet

html_theme = "sphinx_book_theme"
html_theme_path = [sphinx_book_theme.get_html_theme_path()]

project = 'HelixNet'
copyright = '2025, Amr Fahmy'
author = 'Amr Fahmy'
release = helixnet.__version__

extensions = [
    "sphinx.ext.autodoc"
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
html_static_path = ['_static']
