# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import sys
from pathlib import Path

html_theme = "bizstyle"
#html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

project = 'HelixNet'
copyright = '2025, Amr Fahmy'
author = 'Amr Fahmy'
release = '0.1.6'

extensions = [
    "sphinx.ext.autodoc"
]

sys.path.insert(0, str(Path('..', 'src').resolve()))

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
html_static_path = ['_static']
