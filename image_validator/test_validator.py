"""
test_validator.py
==================

Lightweight smoke tests using synthetically generated images (no
external test-image dataset required). These check that each rejection
path fires as expected and that a clean document image is accepted —
they are not a substitute for the real calibration described in the
README, which needs a real labelled dataset.

Run with:
    python -m pytest image_validator/tests/test_validator.py -v
or, without pytest:
    python image_validator/tests/test_validator.py
"""

import os
import sys
import tempfile

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from image_validator import ImageValidator, ValidationConfig


def _make_good_document():
    doc = np.full((1400, 1000, 3), 228, dtype=np.uint8)
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 3, doc.shape).astype(np.int16)
    doc = np.clip(doc.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    for y in range(100, 1300, 40):
        cv2.line(doc, (80, y), (900, y), (30, 30, 30), 2)
        for x in range(100, 900, 30):
            cv2.rectangle(doc, (x, y - 15), (x + 15, y - 5), (20, 20, 20), -1)
    return doc


def _save_tmp(img, suffix=".jpg"):
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    cv2.imwrite(path, img)
    return path


def test_good_document_is_accepted():
    validator = ImageValidator()
    path = _save_tmp(_make_good_document())
    try:
        result = validator.validate(path)
        assert result.valid, result.reasons
        assert result.status == "ACCEPTED"
    finally:
        os.remove(path)


def test_blurry_document_is_rejected():
    validator = ImageValidator()
    blurry = cv2.GaussianBlur(_make_good_document(), (35, 35), 20)
    path = _save_tmp(blurry)
    try:
        result = validator.validate(path)
        assert not result.valid
        assert result.checks["blur"] is False
    finally:
        os.remove(path)


def test_too_small_is_rejected():
    validator = ImageValidator()
    small = cv2.resize(_make_good_document(), (300, 200))
    path = _save_tmp(small)
    try:
        result = validator.validate(path)
        assert not result.valid
        assert result.checks["resolution"] is False
    finally:
        os.remove(path)


def test_too_dark_is_rejected():
    validator = ImageValidator()
    dark = (_make_good_document().astype(np.float32) * 0.12).astype(np.uint8)
    path = _save_tmp(dark)
    try:
        result = validator.validate(path)
        assert not result.valid
        assert result.checks["brightness"] is False
    finally:
        os.remove(path)


def test_too_bright_is_rejected():
    """
    NOTE: brightness/contrast now use a trimmed mean/std (see
    checks._trimmed_values) that excludes fully-saturated pixels before
    computing either statistic — this deliberately lets a genuinely
    overexposed image (background blown fully to white AND surviving
    content washed to low contrast) surface as a CONTRAST failure
    instead of a BRIGHTNESS failure, since once saturated pixels are
    trimmed out, what's left is a narrow, low-variance band rather than
    a high mean. Either check firing is correct here — what matters is
    that the image is still rejected overall — so this test checks
    `result.valid` and either flag, not brightness specifically.
    """
    validator = ImageValidator()
    bright = np.clip(_make_good_document().astype(np.float32) + 180, 0, 255).astype(np.uint8)
    path = _save_tmp(bright)
    try:
        result = validator.validate(path)
        assert not result.valid
        assert result.checks["brightness"] is False or result.checks["contrast"] is False
    finally:
        os.remove(path)


def test_glare_mechanism_detects_localized_bright_spot():
    """
    The DEFAULT config deliberately uses a loose glare threshold (see
    config.py's glare_blob_area_ratio_threshold comment): a pixel-
    brightness heuristic cannot reliably tell real camera glare apart
    from ordinary printed white ink on a label, so by default this
    check only catches extreme cases and otherwise leaves glare_ratio
    as an advisory metric rather than a hard gate — measured, real
    legible product-label photos and a damaging synthetic glare case
    land in overlapping ranges, so a tight default would reject good
    photos. This test therefore checks the underlying MECHANISM (a
    bright spot over text measurably raises glare_ratio, and a
    strict/custom threshold DOES reject it), rather than asserting
    default-config behaviour for a moderate synthetic case.
    """
    doc = _make_good_document()
    cv2.circle(doc, (700, 650), 150, (255, 255, 255), -1)
    doc = cv2.GaussianBlur(doc, (0, 0), sigmaX=3)
    path = _save_tmp(doc)
    try:
        lenient_result = ImageValidator().validate(path)
        assert lenient_result.metrics.glare_ratio > 0.03, lenient_result.metrics.glare_ratio

        strict_cfg = ValidationConfig()
        strict_cfg.glare_blob_area_ratio_threshold = 0.02
        strict_result = ImageValidator(strict_cfg).validate(path)
        assert strict_result.checks["glare"] is False
    finally:
        os.remove(path)


def test_blank_image_is_rejected():
    validator = ImageValidator()
    blank = np.full((1200, 900, 3), 200, dtype=np.uint8)
    path = _save_tmp(blank)
    try:
        result = validator.validate(path)
        assert not result.valid
        assert result.checks["blank"] is False
    finally:
        os.remove(path)


def test_corrupted_file_is_rejected_without_crashing():
    validator = ImageValidator()
    fd, path = tempfile.mkstemp(suffix=".jpg")
    with os.fdopen(fd, "wb") as f:
        f.write(b"not a real image" * 10)
    try:
        result = validator.validate(path)
        assert not result.valid
        assert result.checks["file_valid"] is False
    finally:
        os.remove(path)


def test_missing_file_is_rejected_without_crashing():
    validator = ImageValidator()
    result = validator.validate("/tmp/this_path_does_not_exist_12345.jpg")
    assert not result.valid
    assert result.checks["file_valid"] is False


def test_folder_validation_returns_one_result_per_image():
    validator = ImageValidator()
    tmp_dir = tempfile.mkdtemp()
    try:
        cv2.imwrite(os.path.join(tmp_dir, "a.jpg"), _make_good_document())
        cv2.imwrite(os.path.join(tmp_dir, "b.png"), _make_good_document())
        with open(os.path.join(tmp_dir, "not_an_image.txt"), "w") as f:
            f.write("ignore me")
        results = validator.validate_path(tmp_dir)
        assert len(results) == 2  # the .txt file must be skipped
    finally:
        for name in os.listdir(tmp_dir):
            os.remove(os.path.join(tmp_dir, name))
        os.rmdir(tmp_dir)


if __name__ == "__main__":
    # Allow running without pytest installed.
    tests = [obj for name, obj in list(globals().items()) if name.startswith("test_")]
    passed, failed = 0, 0
    for test in tests:
        try:
            test()
            print(f"PASS: {test.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL: {test.__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
