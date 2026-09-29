"""Pytest bootstrap for the plugin test suite.

The tests import the plugin through its runtime package path
(``data.plugins.astrbot_plugin_novelai``), mirroring how AstrBot loads it.
This file makes that import work when the repository is checked out anywhere,
including a plain ``git clone`` or a GitHub Actions runner.
"""

import sys
from pathlib import Path


def _ensure_astrbot_root_on_path() -> None:
    """Add the directory that contains the ``data/plugins`` tree to sys.path."""

    for parent in Path(__file__).resolve().parents:
        if (parent / "data" / "plugins").is_dir():
            root = str(parent)
            if root not in sys.path:
                sys.path.insert(0, root)
            return


_ensure_astrbot_root_on_path()
