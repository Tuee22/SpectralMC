"""
Importing spectralmc.cvnn registers ~60 test functions it already
contains (they’re defined at module import time).  Pytest will
discover and run them normally.
"""

import spectralmc.cvnn  # noqa: F401  (import side‑effect)
