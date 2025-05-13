# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import sys

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'OligoGym'
copyright = '2025, Carlo De Donno, Rachapun Rotrattanadumrong, RNAHub, pRED, Roche'
author = 'Carlo De Donno, Rachapun Rotrattanadumrong, RNAHub, pRED, Roche'
release = '1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.intersphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "autoapi.extension",
    "myst_nb",
]

source_suffix = ['.rst', '.ipynb', '.md']
autoapi_dirs = ['../oligogym/']
autoapi_type = 'python'
autodoc_typehints = "description"
napoleon_custom_sections = [('Returns', 'params_style')]
nb_execution_mode = "off"
nb_output_stderr = "remove"
nb_merge_streams = True
myst_render_markdown_format = "myst"
master_doc = 'index'
templates_path = ['_templates']
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", ".venv", "**/.ipynb_checkpoints"]
nitpick_ignore = [('py:class', 'type')]
pygments_style = "pastie"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']

# -- External mapping --------------------------------------------------------
python_version = ".".join(map(str, sys.version_info[0:2]))
intersphinx_mapping = {
    "sphinx": ("https://www.sphinx-doc.org/en/master", None),
    "python": ("https://docs.python.org/" + python_version, None),
    "matplotlib": ("https://matplotlib.org", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "sklearn": ("https://scikit-learn.org/stable", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference", None),
    "ipython": ("https://ipython.readthedocs.io/en/stable/", None),

}

print(f"loading configurations for {project} ...", file=sys.stderr)