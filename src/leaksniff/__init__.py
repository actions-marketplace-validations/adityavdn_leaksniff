"""leaksniff: catch data leakage and evaluation mistakes in ML code before you report a score."""
from .core import Finding, check_path, check_source

__all__ = ["Finding", "check_path", "check_source"]
__version__ = "0.1.0"
