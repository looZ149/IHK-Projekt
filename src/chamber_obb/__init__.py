"""chamber_obb library core -- installable OBB detection pipeline (CAP-2, CAP-4).

Exposes exactly one public entry point, ``analyze()``, plus the ``FrameMetadata``
payload type it accepts (AD-2). ``preprocessing``/``inference``/``postprocessing``
are internal implementation detail and are not re-exported here.
"""

from chamber_obb.pipeline import FrameMetadata, analyze

__all__ = ["FrameMetadata", "analyze"]
