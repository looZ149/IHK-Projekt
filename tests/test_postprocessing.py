"""Stage-boundary Test fuer chamber_obb.postprocessing (AD-14, Story 4).

Deckt den Letterbox-Inverse-Transform end-to-end ab: ein nicht-quadratisches,
nicht-640er Originalbild wird via Story 2s ``letterbox_resize`` in den
640x640-Raum gebracht, und Story 4s ``invert_letterbox_coords`` muss die
Original-Pixel-Koordinaten exakt zurueckliefern (AD-14's eigene Testbeschreibung).
Kein echtes Modell/GPU noetig -- die Rueckwaertstransformation ist rein
mathematisch testbar.
"""

import numpy as np
import pytest

from chamber_obb.postprocessing import invert_letterbox_coords, postprocess
from chamber_obb.preprocessing import LetterboxInfo, letterbox_resize

from fakes import FakeOBB, FakeResults


def test_invert_letterbox_coords_nonsquare_roundtrip():
    """1200x600 Original -> 640x640 letterboxed -> zurueck in Original-Raum."""
    orig = np.zeros((600, 1200, 3), dtype=np.uint8)
    _, info = letterbox_resize(orig, target_size=640)

    # scale = min(640/1200, 640/600) = 640/1200 = 8/15; new_width=640, new_height=320
    # -> pad_left=0, pad_top=(640-320)//2=160
    assert info.pad_left == 0
    assert info.pad_top == 160
    assert info.scale == pytest.approx(8 / 15)

    # Punkt innerhalb des unpadded Bereichs (y=200 liegt zwischen pad_top=160
    # und pad_top+new_height=480).
    letterboxed_point = np.array([320.0, 200.0])
    inverted = invert_letterbox_coords(letterboxed_point, info)

    # Handgerechnet: x = 320 / (8/15) = 600.0, y = (200-160) / (8/15) = 75.0
    assert inverted == pytest.approx([600.0, 75.0])
    assert 0 <= inverted[0] <= info.orig_width
    assert 0 <= inverted[1] <= info.orig_height


def test_invert_letterbox_coords_portrait_roundtrip():
    """600x1200 (portrait) Original -> 640x640 letterboxed -> zurueck in Original-Raum.

    Spiegelbild von test_invert_letterbox_coords_nonsquare_roundtrip: hier wird
    die Breite gepolstert (pad_left>0) statt der Hoehe (pad_top>0).
    """
    orig = np.zeros((1200, 600, 3), dtype=np.uint8)
    _, info = letterbox_resize(orig, target_size=640)

    # scale = min(640/600, 640/1200) = 640/1200 = 8/15; new_width=320, new_height=640
    # -> pad_left=(640-320)//2=160, pad_top=0
    assert info.pad_left == 160
    assert info.pad_top == 0
    assert info.scale == pytest.approx(8 / 15)

    # Punkt innerhalb des unpadded Bereichs (x=200 liegt zwischen pad_left=160
    # und pad_left+new_width=480).
    letterboxed_point = np.array([200.0, 320.0])
    inverted = invert_letterbox_coords(letterboxed_point, info)

    # Handgerechnet: x = (200-160) / (8/15) = 75.0, y = 320 / (8/15) = 600.0
    assert inverted == pytest.approx([75.0, 600.0])
    assert 0 <= inverted[0] <= info.orig_width
    assert 0 <= inverted[1] <= info.orig_height


def test_invert_letterbox_coords_works_on_axis_aligned_box_shape():
    """Funktioniert uniform auch auf (2, 2)-achsparallelen Boxen (zwei Eckpunkte)."""
    info = LetterboxInfo(
        scale=2.0,
        pad_left=10,
        pad_top=20,
        orig_width=100,
        orig_height=100,
        new_width=200,
        new_height=200,
    )
    box = np.array([[10.0, 20.0], [30.0, 40.0]])  # (2, 2): [x1,y1], [x2,y2]
    inverted = invert_letterbox_coords(box, info)
    expected = np.array([[0.0, 0.0], [10.0, 10.0]])
    assert inverted == pytest.approx(expected)


def test_invert_letterbox_coords_padding_region():
    """Ein Eckpunkt im Padding-Band invertiert bewusst NICHT geclippt nach ausserhalb."""
    orig = np.zeros((600, 1200, 3), dtype=np.uint8)
    _, info = letterbox_resize(orig, target_size=640)

    # y=50 liegt oberhalb von pad_top=160 -> im oberen Padding-Band.
    letterboxed_point = np.array([100.0, 50.0])
    inverted = invert_letterbox_coords(letterboxed_point, info)

    assert inverted[1] < 0 or inverted[1] > info.orig_height


def test_invert_letterbox_coords_no_padding_is_identity():
    """scale=1.0, pad_left=pad_top=0 -> invertierte Koordinaten == Eingabe."""
    info = LetterboxInfo(
        scale=1.0,
        pad_left=0,
        pad_top=0,
        orig_width=640,
        orig_height=640,
        new_width=640,
        new_height=640,
    )
    coords = np.array([[10.0, 20.0], [300.5, 400.25]])
    inverted = invert_letterbox_coords(coords, info)
    assert inverted == pytest.approx(coords)


def test_invert_letterbox_coords_works_on_obb_corner_shape():
    """Funktioniert uniform auf (N, 4, 2)-OBB-Eckpunkten, nicht nur Einzelpunkten."""
    info = LetterboxInfo(
        scale=2.0,
        pad_left=10,
        pad_top=20,
        orig_width=100,
        orig_height=100,
        new_width=200,
        new_height=200,
    )
    corners = np.array([[[10.0, 20.0], [30.0, 20.0], [30.0, 40.0], [10.0, 40.0]]])
    inverted = invert_letterbox_coords(corners, info)
    expected = np.array([[[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]])
    assert inverted == pytest.approx(expected)


def test_postprocess_zero_detections_returns_empty_list():
    fake_results = FakeResults(
        obb=FakeOBB(np.zeros((0, 4, 2)), np.zeros((0,)), np.zeros((0,))),
        names={0: "chamber_region", 1: "channel_region"},
    )
    info = LetterboxInfo(
        scale=1.0, pad_left=0, pad_top=0, orig_width=640, orig_height=640, new_width=640, new_height=640
    )
    assert postprocess(fake_results, info) == []


def test_postprocess_builds_detections_from_results_names():
    """Klassennamen kommen aus ``results.names`` -- nie hartkodiert (AD-13)."""
    corners = np.array(
        [[[0.0, 160.0], [320.0, 160.0], [320.0, 480.0], [0.0, 480.0]]]
    )  # ein Detection-Box in 640-Raum
    conf = np.array([0.91])
    cls = np.array([1])
    fake_results = FakeResults(
        obb=FakeOBB(corners, conf, cls),
        names={0: "chamber_region", 1: "channel_region"},
    )
    orig = np.zeros((600, 1200, 3), dtype=np.uint8)
    _, info = letterbox_resize(orig, target_size=640)

    detections = postprocess(fake_results, info)

    assert len(detections) == 1
    det = detections[0]
    assert det.cls_id == 1
    assert det.cls_name == "channel_region"
    assert det.confidence == pytest.approx(0.91)
    assert det.corners == pytest.approx(
        np.array([[0.0, 0.0], [600.0, 0.0], [600.0, 600.0], [0.0, 600.0]])
    )


def test_postprocess_keeps_multiple_same_class_detections():
    """CAP-2: keine Best-of-Class-Filterung -- zwei chamber_region-Boxen bleiben beide erhalten."""
    corners = np.array(
        [
            [[0.0, 160.0], [100.0, 160.0], [100.0, 260.0], [0.0, 260.0]],
            [[200.0, 160.0], [300.0, 160.0], [300.0, 260.0], [200.0, 260.0]],
        ]
    )
    conf = np.array([0.85, 0.77])
    cls = np.array([0, 0])  # beide dieselbe Klasse
    fake_results = FakeResults(
        obb=FakeOBB(corners, conf, cls),
        names={0: "chamber_region", 1: "channel_region"},
    )
    info = LetterboxInfo(
        scale=1.0, pad_left=0, pad_top=0, orig_width=640, orig_height=640, new_width=640, new_height=640
    )

    detections = postprocess(fake_results, info)

    assert len(detections) == 2
    assert [d.cls_name for d in detections] == ["chamber_region", "chamber_region"]
    assert [d.confidence for d in detections] == pytest.approx([0.85, 0.77])
