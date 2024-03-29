#
# Copyright (c) 2020 LA EPFL.
#
# This file is part of MPOPT
# (see http://github.com/mpopt).
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#
# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Jupyter documentation build configuration file.
#
# This file is execfile()d with the current directory set to its
# containing dir.
#
import sys
import os
import shlex

sys.path.insert(0, os.path.abspath("../.."))
sys.setrecursionlimit(1500)

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
# sys.path.insert(0, os.path.abspath('.'))

# -- General configuration ------------------------------------------------

html_theme = "pydata_sphinx_theme"
# html_theme = "sphinx_rtd_theme"
html_logo = "_static/_images/mpopt.svg"
html_favicon = "_static/_images/favicon.jpg"

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    # 'sphinx.ext.graphviz', # Add the graphviz extension
    # 'sphinxext.rediraffe',
    "myst_parser",
    "sphinx_design",
    "sphinx.ext.doctest",
    "sphinx.ext.todo",
    "sphinx.ext.mathjax",
    "sphinx.ext.ifconfig",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "nbsphinx",
]

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = True
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True

# panels_add_bootstrap_css = False

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
source_suffix = [".rst", ".md"]

# The encoding of source files.
# source_encoding = 'utf-8-sig'

# The master toctree document.
master_doc = "index"


# -- Project information -----------------------------------------------------
# General information about the project.
project = "mpopt"
copyright = "2023, Devakumar THAMMISETTY, Colin Jones"
author = "Devakumar THAMMISETTY, Colin Jones"

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.
version = "0"
# The full version, including alpha/beta/rc tags
release = "0.2.2"

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = "en"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ["_build"]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False

# -- Options for HTML output ----------------------------------------------

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

html_context = {
    # "github_url": "https://github.com", # or your GitHub Enterprise interprise
    "github_user": "mpopt",
    "github_repo": "mpopt",
    "github_version": "docs",
    "doc_path": "docs/source",
}

html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/mpopt/mpopt",
            "icon": "fab fa-github-square",
        },
    ],
    "external_links": [
        # {"name": "mpopt.org", "url": "https://mpopt.org"},
    ],
    "use_edit_page_button": False,
}

# If not '', a 'Last updated on:' timestamp is inserted at every page bottom,
# using the given strftime format.
html_last_updated_fmt = "%Y-%m-%d"

# Custom sidebar templates, maps document names to template names.
# html_sidebars = {}

# Additional templates that should be rendered to pages, maps page names to
# template names.
html_additional_pages = {}

# Output file base name for HTML help builder.
htmlhelp_basename = "mpopt"

# -- Options for LaTeX output ---------------------------------------------

latex_elements = {}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (
        master_doc,
        "mpopt.tex",
        "MPOPT Documentation",
        "https://mpopt.readthedocs.io/",
        "manual",
    ),
]

# -- Options for manual page output ---------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [(master_doc, "mpopt", "MPOPT Documentation", [author], 1)]

# -- Options for Texinfo output -------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (
        master_doc,
        "mpopt",
        "MPOPT Documentation",
        author,
        "mpopt",
        "One line description of project.",
        "Miscellaneous",
    ),
]


# -- Options for intersphinx -----------------------------------------------

intersphinx_mapping = {
    "mpopt": ("https://mpopt.readthedocs.io/en/latest/", None),
}

intersphinx_cache_limit = 5

# -- Translation ----------------------------------------------------------

gettext_uuid = True
locale_dirs = ["locale/"]


def setup(app):
    app.add_css_file("custom.css")
