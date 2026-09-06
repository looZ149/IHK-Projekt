import numpy as np
import tifffile


"""def normalize_to_8bit(img, p_low=1, p_high=99):
    lo, hi = np.percentile(img, [p_low, p_high])
    clipped = np.clip(img, lo, hi)
    return ((clipped - lo) / (hi - lo + 1e-8) * 255).astype(np.uint8)"""

def normalize_to_8bit(img: np.ndarray, p_low: float = 1, p_high: float = 99) -> np.ndarray:
    """Skaliert ein Bild per Perzentil-Clipping auf 8-bit.

    Robuster als Min-Max-Normalisierung gegenueber Ausreisser-Pixeln und
    besser geeignet fuer den engen Wertebereich von Phasenkontrast-
    Mikroskopiebildern.
    """
    lo, hi = np.percentile(img, [p_low, p_high])
    clipped = np.clip(img, lo, hi)
    return ((clipped - lo) / (hi - lo + 1e-8) * 255).astype(np.uint8)


def load_and_normalize(path) -> np.ndarray:
    img = tifffile.imread(path)
    with tifffile.TiffFile(path) as tif:
        midFrame = len(tif.pages)
        midFrame = midFrame / 2
        midFrame = int(midFrame)

    # Grab some mid frame for now
    if img.ndim > 2:
        img = img[midFrame]

    # Convert to 8 Bit (Perzentil-Clipping statt Min-Max, siehe Grund unten)
    if img.dtype != np.uint8:
        img = normalize_to_8bit(img)

    if img.ndim != 2:
        img = np.stack([img] * 3, axis=-1)

    return img