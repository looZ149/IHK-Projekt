import logging
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
import tifffile

logger = logging.getLogger(__name__)


def normalize_to_8bit(img: np.ndarray, p_low: float = 1, p_high: float = 99) -> np.ndarray:
    """Skaliert ein Bild per Perzentil-Clipping auf 8-bit.

    Robuster als Min-Max-Normalisierung gegenueber Ausreisser-Pixeln und
    besser geeignet fuer den engen Wertebereich von Phasenkontrast-
    Mikroskopiebildern.
    """
    lo, hi = np.percentile(img, [p_low, p_high])
    clipped = np.clip(img, lo, hi)
    return ((clipped - lo) / (hi - lo + 1e-8) * 255).astype(np.uint8)


class LetterboxInfo(NamedTuple):
    """Skalierungs-/Padding-Metadaten, die ``letterbox_resize`` liefert.

    Wird an den Aufrufer zurueckgegeben (hier nicht verbraucht/verworfen),
    damit eine spaetere Stufe Koordinaten zurueck in den urspruenglichen
    Pixel-Raum transformieren kann (AD-14).
    """

    scale: float
    pad_left: int
    pad_top: int
    orig_width: int
    orig_height: int
    new_width: int
    new_height: int


def letterbox_resize(
    image: np.ndarray, target_size: int = 640, fill_value: int = 114
) -> tuple[np.ndarray, LetterboxInfo]:
    """Resiziert ``image`` auf ``target_size``x``target_size`` ohne Beschneidung.

    Skaliert gleichmaessig auf beiden Achsen (Seitenverhaeltnis bleibt
    erhalten, sowohl Up- als auch Downscale) und fuellt die kuerzere Achse
    mit ``fill_value`` auf, statt zuzuschneiden.
    """
    if target_size <= 0:
        raise ValueError(f"target_size must be positive, got {target_size}")
    if image.ndim < 2:
        raise ValueError(f"image must have at least 2 dimensions, got shape {image.shape}")

    orig_height, orig_width = image.shape[:2]
    if orig_width <= 0 or orig_height <= 0:
        raise ValueError(f"image has a zero-length dimension: {image.shape}")

    if np.issubdtype(image.dtype, np.integer):
        dtype_range = np.iinfo(image.dtype)
        if not (dtype_range.min <= fill_value <= dtype_range.max):
            raise ValueError(
                f"fill_value {fill_value} is out of range for dtype {image.dtype} "
                f"({dtype_range.min}..{dtype_range.max})"
            )

    scale = min(target_size / orig_width, target_size / orig_height)
    new_width = round(orig_width * scale)
    new_height = round(orig_height * scale)
    if new_width <= 0 or new_height <= 0:
        raise ValueError(
            f"scale {scale} on image {orig_width}x{orig_height} rounds a target "
            f"dimension to zero (new_width={new_width}, new_height={new_height})"
        )

    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    canvas_shape = (target_size, target_size) + image.shape[2:]
    canvas = np.full(canvas_shape, fill_value, dtype=image.dtype)

    pad_left = (target_size - new_width) // 2
    pad_top = (target_size - new_height) // 2
    canvas[pad_top : pad_top + new_height, pad_left : pad_left + new_width] = resized

    info = LetterboxInfo(
        scale=scale,
        pad_left=pad_left,
        pad_top=pad_top,
        orig_width=orig_width,
        orig_height=orig_height,
        new_width=new_width,
        new_height=new_height,
    )
    return canvas, info


def load_and_normalize(path) -> np.ndarray:
    """Laedt ein TIFF, waehlt den mittleren Frame und normalisiert zu 8-bit/3-Kanal.

    Raises:
        ValueError: wenn ``path`` kein str/Path ist, der Pfad nicht existiert,
            oder die Datei leer/korrupt ist und nicht als TIFF gelesen werden kann.
    """
    if not isinstance(path, (str, Path)):
        logger.error("load_and_normalize: path is not a str or Path: %r", path)
        raise ValueError(f"path must be a str or Path, got {type(path).__name__}")

    if not Path(path).is_file():
        logger.error("load_and_normalize: file not found: %s", path)
        raise ValueError(f"Image file not found: {path}")

    try:
        img = tifffile.imread(path)
        with tifffile.TiffFile(path) as tif:
            midFrame = len(tif.pages)
            midFrame = midFrame / 2
            midFrame = int(midFrame)
    except Exception as exc:
        logger.error("load_and_normalize: failed to decode TIFF %s: %s", path, exc)
        raise ValueError(f"Could not read TIFF file: {path}") from exc

    # Grab some mid frame for now
    if img.ndim > 2:
        img = img[midFrame]

    # Convert to 8 Bit (Perzentil-Clipping statt Min-Max, siehe Grund unten)
    if img.dtype != np.uint8:
        img = normalize_to_8bit(img)

    if img.ndim != 2:
        img = np.stack([img] * 3, axis=-1)

    return img


def preprocess(path) -> tuple[np.ndarray, LetterboxInfo]:
    """Stage-1 entry point: laedt+normalisiert ein TIFF und letterboxt es auf 640x640."""
    img = load_and_normalize(path)
    return letterbox_resize(img)
