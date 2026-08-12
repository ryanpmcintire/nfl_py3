"""NFL against-the-spread research and prediction tools."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("nfl-ats")
except PackageNotFoundError:  # pragma: no cover - editable installs provide metadata
    __version__ = "0.0.0"

__all__ = ["__version__"]
