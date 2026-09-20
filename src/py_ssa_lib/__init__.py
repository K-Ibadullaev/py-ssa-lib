from importlib.metadata import PackageNotFoundError
from importlib.metadata import version 

try:
    __version__ = version("py_ssa_lib")
except PackageNotFoundError:
    __version__ = "Package not found"