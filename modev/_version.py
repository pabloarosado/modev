import pkg_resources
# Load version given in setup.py.
__version__ = pkg_resources.require("modev")[0].version
