"""Geteilte Test-Doubles fuer die Inferenz-/Postprocessing-Stufen.

Minimaler Stand-in fuer ultralytics' ``Results``/``Results.obb`` -- kein
Modell, keine GPU. Von ``test_pipeline.py`` und ``test_postprocessing.py``
gemeinsam genutzt (dort zuvor als identische Kopien).
"""


class FakeOBB:
    """Minimaler Stand-in fuer ultralytics' ``Results.obb`` -- kein Modell/GPU noetig."""

    def __init__(self, xyxyxyxy, conf, cls):
        self.xyxyxyxy = xyxyxyxy
        self.conf = conf
        self.cls = cls

    def __len__(self):
        return len(self.cls)


class FakeResults:
    def __init__(self, obb, names):
        self.obb = obb
        self.names = names
