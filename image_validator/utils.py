"""
utils.py
========

Small shared helpers used by multiple checks. Keeping these separate
avoids duplicating "load and downscale" logic inside every check
function, and gives us exactly one place that touches disk I/O.
"""

import os
import time
from contextlib import contextmanager
from typing import Optional, Tuple

import cv2
import numpy as np

from image_validator.config import ValidationConfig


def is_supported_file(path: str, cfg: ValidationConfig) -> bool:
    """Cheap extension check before we even try to touch the file."""
    ext = os.path.splitext(path)[1].lower()
    return ext in cfg.supported_extensions


def safe_imread(path: str) -> Optional[np.ndarray]:
    """
    Attempt to decode an image file into a BGR numpy array.

    Returns None (never raises) on any failure — missing file, zero-byte
    file, truncated/corrupted image, unsupported codec, etc. This is the
    single point where "can this even be decoded" is decided, so
    check_file_validity() and every downstream check can assume that a
    non-None array is a genuinely readable image.

    cv2.imread() itself never raises on a bad file — it just returns
    None — but we still wrap it in try/except because some corrupted
    files (truncated headers, hostile/malformed bytes) can raise inside
    OpenCV's underlying decoders rather than returning None cleanly.
    """
    if not os.path.isfile(path):
        return None
    try:
        # IMREAD_COLOR normalizes everything (including grayscale PNGs,
        # RGBA PNGs, etc.) to a 3-channel BGR array so every downstream
        # check can rely on a consistent shape.
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


def downscale_for_analysis(img: np.ndarray, max_dimension: int) -> np.ndarray:
    """
    Return a resized copy for quality analysis (blur/brightness/contrast/
    glare/blank/relevance), capped at `max_dimension` on the longer side.

    Why: variance-of-Laplacian, mean/std of intensity, and edge-density
    are all statistical measures over pixel distributions. Past a few
    hundred thousand pixels, additional resolution changes these
    statistics negligibly but costs real CPU time — a 4000x3000 phone
    photo is ~12M pixels vs. ~750K after capping to 1000px, i.e. a ~16x
    reduction in work for every downstream check, for a result that is
    numerically almost identical. Resolution itself is still measured
    against the ORIGINAL (undownscaled) image — see check_resolution.
    """
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest <= max_dimension:
        return img
    scale = max_dimension / float(longest)
    new_w, new_h = int(w * scale), int(h * scale)
    # INTER_AREA is the recommended OpenCV interpolation for shrinking
    # images — it avoids the moire/aliasing artifacts that INTER_LINEAR
    # can introduce when downsampling, which would otherwise distort the
    # very texture/edge statistics we're about to measure.
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def to_gray(img: np.ndarray) -> np.ndarray:
    """Convert BGR -> grayscale once; several checks share this."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


@contextmanager
def timer():
    """
    Simple context manager to measure elapsed wall-clock time in
    milliseconds, used to populate `validation_time_ms` in the result.

        with timer() as t:
            do_work()
        print(t.elapsed_ms)
    """
    class _Timer:
        elapsed_ms: float = 0.0

    t = _Timer()
    start = time.perf_counter()
    try:
        yield t
    finally:
        t.elapsed_ms = (time.perf_counter() - start) * 1000.0


def discover_images(path: str, cfg: ValidationConfig):
    """
    Given either a single file path or a directory, return a sorted list
    of file paths to validate. Directories are scanned non-recursively
    (upload folders are expected to be flat, per the integration
    assumption in the spec) and filtered to supported extensions.
    """
    if os.path.isdir(path):
        entries = []
        for name in sorted(os.listdir(path)):
            full = os.path.join(path, name)
            if os.path.isfile(full) and is_supported_file(full, cfg):
                entries.append(full)
        return entries
    return [path]
