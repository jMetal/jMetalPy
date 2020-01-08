# -- Path setup --------------------------------------------------------------

import os
import sys
sys.path.insert(0, os.path.abspath('../../'))

# -- Project information -----------------------------------------------------

project = 'jMetalPy'
copyright = '2019, Antonio Benítez-Hidalgo'
author = 'Antonio Benítez-Hidalgo'
version = ''
release = '1.5.3'


# -- General configuration ---------------------------------------------------

# needs_sphinx = '1.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.ifconfig',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'nbsphinx'
]

templates_path = ['_templates']
source_suffix = '.rst'
master_doc = 'index'
language = None
pygments_style = 'sphinx'

# -- Options for HTML output -------------------------------------------------

html_show_sourcelink = False
html_theme_path = templates_path
html_theme = 'guzzle'
html_css_files = ['custom.css']
html_theme_options = {
    "project_nav_name": "Python version of the jMetal framework",
    "project_nav_logo": "_static/jmetalpy.png"
}
html_static_path = ['_static']
html_sidebars = {
    '**': ['logo-text.html',
           'globaltoc.html',
           'localtoc.html',
           'searchbox.html']
}

# -- Options for HTMLHelp output ---------------------------------------------

htmlhelp_basename = 'jMetalPy'

# -- Options for Sphinx output -------------------------------------------------

exclude_patterns = ['_build', '**.ipynb_checkpoints']

nbsphinx_execute_arguments = [
    "--InlineBackend.figure_formats={'svg', 'pdf'}",
    "--InlineBackend.rc={'figure.dpi': 96}",
]
nbsphinx_execute = 'never'

# -- Options for LaTeX output ------------------------------------------------

latex_elements = {
}

latex_documents = [
    (master_doc, 'jMetalPy.tex', 'jMetalPy Documentation',
     'Antonio Benítez-Hidalgo', 'manual'),
]

# -- Options for manual page output ------------------------------------------

man_pages = [
    (master_doc, 'jmetalpy', 'jMetalPy Documentation',
     [author], 1)
]

# -- Options for Texinfo output ----------------------------------------------

texinfo_documents = [
    (master_doc, 'jMetalPy', 'jMetalPy Documentation',
     author, 'jMetalPy', 'Python version of the jMetal framework.',
     'Miscellaneous'),
]

# -- Extension configuration -------------------------------------------------

# -- Options for intersphinx extension ---------------------------------------

intersphinx_mapping = {'https://docs.python.org/': None}