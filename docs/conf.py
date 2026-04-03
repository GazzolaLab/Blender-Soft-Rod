# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

sys.path.insert(0, os.path.abspath("../src"))
from bsr import version

project = "SPARC"
copyright = "Seung Hyun Kim"
author = "Seung Hyun Kim"
release = version

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.viewcode",
    "myst_parser",
    "sphinxcontrib.mermaid",
    "sphinx_copybutton",
    "sphinx_autodoc_typehints",
    "sphinx_click",
    "numpydoc",
]


# -- Options for autodoc -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#module-sphinx.ext.autodoc
# List class/module members in source order (default is alphabetical).
autodoc_member_order = "bysource"
autodoc_default_flags = [
    "members",
    "undoc-members",
    "private-members",
    "special-members",
    "inherited-members",
]
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

templates_path = ["_templates"]
exclude_patterns = []
myst_fence_as_directive = ["mermaid"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_context = {
    "github_repo_url": "https://github.com/GazzolaLab/Blender-Soft-Rod/tree/meta2quest-integration",
}

copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True

# -- Numpydoc settings -------------------------------------------------------
numpydoc_show_class_members = False
