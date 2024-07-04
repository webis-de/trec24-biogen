from pathlib import Path
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("trec-biogen")
except PackageNotFoundError:
    pass

PROJECT_DIR = Path(__file__).parent.parent
