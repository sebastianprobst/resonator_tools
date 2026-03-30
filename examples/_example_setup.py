from pathlib import Path
import sys


def ensure_repo_root_on_path() -> Path:
    """Add the repository root to sys.path so local imports work without install."""
    repo_root = Path(__file__).resolve().parents[1]
    root_str = str(repo_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return repo_root


def data_path(filename: str) -> Path:
    """Resolve files relative to the examples directory."""
    return Path(__file__).resolve().parent / filename
