from pathlib import Path
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("trec-biogen")
except PackageNotFoundError:
    __version__ = "0.0.0"

PROJECT_DIR = Path(__file__).parent.parent
