import os
import sys

# `uv sync` installs the package editable, but keep the tests runnable from a
# bare checkout too (e.g. a pod that synced the source but not the venv).
SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
