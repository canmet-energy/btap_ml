# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('../../src'))
sys.path.insert(0, os.path.abspath('../..'))
sys.path.insert(0, os.path.abspath('..'))  
#sys.path.insert(0, os.path.abspath('.'))
sys.setrecursionlimit(1500)

print('bukola')
print(sys.path)
# -- Project information -----------------------------------------------------

project = 'Building Technology Assessment Platform: Machine Learning Implementation'
copyright = '2021, Natural Resources Canada'
author = 'Natural Resources Canada, Statistics Canada'

# The full version, including alpha/beta/rc tags
release = '0.0.1'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.graphviz",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "rinoh.frontend.sphinx",
    "sphinx_autodoc_typehints",   # must be loaded after napoleon
]



# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

autosummary_generate = True
#numpydoc_show_class_members = False
# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'alabaster'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

latex_elements={
    'papersize':'leterpaper',
    'pointsize':'10pt',
    'preamble':'',
    'figure_align':'htbp'
}