from __future__ import annotations

import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_PATH))


project = "xyz"
author = "xyz contributors"
copyright = "2026, xyz contributors"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
]

templates_path = ["_templates"]
exclude_patterns: list[str] = ["api/generated/*"]
suppress_warnings = ["autodoc"]

autoclass_content = "class"
autodoc_member_order = "bysource"

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "sklearn": ("https://scikit-learn.org/stable", None),
}

html_theme = "pydata_sphinx_theme"
html_title = "xyz documentation"
html_static_path = ["_static"]
html_theme_options = {
    "navigation_with_keys": True,
    "show_prev_next": False,
    "header_links_before_dropdown": 6,
}
