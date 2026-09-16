"""
checks.py
=========

Each function here implements exactly one pre-OCR quality check and
returns a small, uniform result: (passed: bool, metric_value, reason: str)

reason is only meaningful when passed=False — it's a short, still fairly
technical description used internally / in logs. The fully user-facing
copy (no "Laplacian", no "HSV") lives in messages.py, generated from
`checks` + `reasons`, not from these raw strings directly.

All checks operate on a pre-downscaled analysis image (see
utils.downscale_for_analysis) except check_resolution, which deliberately
uses the ORIGINAL image's dimensions.
"""

from typing import Tuple, Optional

import cv2
import numpy as np

from image_validator.config import ValidationConfig


def _trimmed_values(gray: np.ndarray, cfg: ValidationConfig) -> np.ndarray:
    """
    Shared helper for check_brightness / check_contrast: returns the
    grayscale pixel values with near-black and near-white pixels
    excluded (see config.trim_low / trim_high). Falls back to the full
    (untrimmed) array in the pathological case where trimming would
    remove every pixel (e.g. a genuinely all-black or all-white frame —
    those should still be scored, just via the untrimmed stats, rather
    than dividing by zero).
    """
    mask = (gray > cfg.trim_low) & (gray < cfg.trim_high)
    trimmed = gray[mask]
    return trimmed if trimmed.size > 0 else gray.reshape(-1)


# ---------------------------------------------------------------------
# 1. File validity
# ---------------------------------------------------------------------
def check_file_validity(img: Optional[np.ndarray]) -> Tuple[bool, str]:
    """
    Confirm the file could actually be decoded into a real image array.

    Why before OCR: every downstream step assumes a valid pixel grid.
    A corrupted/truncated file or an unsupported codec must be caught
    here with a clean rejection, rather than crashing (or silently
    producing garbage) deeper in the OCR pipeline.
    """
    if img is None:
        return False, "File could not be read as a valid image."
    if img.size == 0 or img.shape[0] == 0 or img.shape[1] == 0:
        return False, "Image file is empty or has zero dimensions."
    return True, ""


# ---------------------------------------------------------------------
# 2. Resolution
# ---------------------------------------------------------------------
def check_resolution(
    img: np.ndarray, cfg: ValidationConfig
) -> Tuple[bool, int, int, str]:
    """
    Reject images whose raw pixel dimensions are too small to plausibly
    contain legible text at typical document-capture distance.

    Metric: width and height in pixels, measured on the ORIGINAL image
    (not the downscaled analysis copy) — resolution is the one property
    that downscaling would corrupt the measurement of.

    Why it matters for OCR: character segmentation and recognition need
    a minimum number of pixels per character stroke. Very low-resolution
    captures produce blocky, ambiguous glyphs regardless of focus or
    lighting.
    """
    h, w = img.shape[:2]
    if w < cfg.min_width_px or h < cfg.min_height_px:
        return False, w, h, (
            f"Resolution {w}x{h} is below the minimum "
            f"{cfg.min_width_px}x{cfg.min_height_px}."
        )
    return True, w, h, ""


# ---------------------------------------------------------------------
# 3. Blur
# ---------------------------------------------------------------------
def check_blur(gray: np.ndarray, cfg: ValidationConfig) -> Tuple[bool, float, str]:
    """
    Metric: variance of the Laplacian.

    The Laplacian is a second-derivative (edge-detecting) operator. A
    sharp image has many crisp intensity transitions (text edges, line
    borders) which the Laplacian responds to strongly and in varied
    ways -> high variance. A blurred image smooths those transitions
    out -> the Laplacian response is weak and uniform -> low variance.

    Why it matters for OCR: OCR engines segment characters based on
    edge/stroke boundaries. Blur destroys exactly that information, so
    this is one of the most direct predictors of OCR failure among all
    the checks here.

    Threshold: see config.py — this is a widely used starting heuristic,
    not a formal standard, and should be calibrated on real data.
    """
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if blur_score < cfg.blur_laplacian_threshold:
        return False, blur_score, "Image is too blurry for reliable text recognition."
    return True, blur_score, ""


# ---------------------------------------------------------------------
# 4. Brightness / exposure
# ---------------------------------------------------------------------
def check_brightness(gray: np.ndarray, cfg: ValidationConfig) -> Tuple[bool, float, str]:
    """
    Metric: mean grayscale pixel intensity (0 = black, 255 = white),
    computed on a TRIMMED pixel set that excludes near-fully-saturated
    pixels (very near black or very near white) before averaging.

    Why trimmed and not a plain full-frame mean: uploads are real-world
    photos of packaging/labels, not flat document scans — the frame
    can contain a plain white studio backdrop, a dark shadowed table, a
    blown-out window, a black background, etc., depending entirely on
    where/how the officer or user took the photo. None of that is
    something we can assume in advance (no "the background is always
    white" assumption), but in every case those saturated pixels are
    uninformative about whether the printed text itself is legible —
    they pull a naive whole-frame mean toward extreme values regardless
    of how well-exposed the actual content is. Trimming them out keeps
    the metric focused on the tonal range that could plausibly contain
    text, on any background.

    Why it matters for OCR: text/background separation depends on
    visible tonal range. A severely underexposed image compresses
    everything toward black (text and background become indistinguishable
    dark blobs); a severely overexposed image clips everything toward
    white (text detail is blown out and lost), in both cases before OCR
    ever runs.
    """
    values = _trimmed_values(gray, cfg)
    brightness_score = float(np.mean(values))
    if brightness_score < cfg.brightness_min:
        return False, brightness_score, "Image is too dark."
    if brightness_score > cfg.brightness_max:
        return False, brightness_score, "Image is overexposed / too bright."
    return True, brightness_score, ""


# ---------------------------------------------------------------------
# 5. Contrast
# ---------------------------------------------------------------------
def check_contrast(gray: np.ndarray, cfg: ValidationConfig) -> Tuple[bool, float, str]:
    """
    Metric: standard deviation of grayscale pixel intensities, computed
    on the same trimmed pixel set as check_brightness (see its docstring
    for why: an arbitrary, unpredictable real-world background — not
    necessarily white — shouldn't be allowed to dominate the statistic).

    A document with clearly visible dark text on a light background
    spans a wide range of intensity values -> high std dev. A washed-out
    or extremely flat-lit capture clusters tightly around a single
    intensity -> low std dev, meaning text and background are close to
    indistinguishable even if nothing is technically over/under-exposed.

    This is a deliberately low, permissive floor — the intent is to
    catch clearly unreadable captures, not to penalize ordinary
    documents with naturally modest contrast.
    """
    values = _trimmed_values(gray, cfg)
    contrast_score = float(np.std(values))
    if contrast_score < cfg.contrast_std_min:
        return False, contrast_score, "Image has very low contrast; text may not be visible."
    return True, contrast_score, ""


# ---------------------------------------------------------------------
# 6. Glare / reflection
# ---------------------------------------------------------------------
def check_glare(
    bgr: np.ndarray, cfg: ValidationConfig, text_mask: Optional[np.ndarray] = None
) -> Tuple[bool, float, str]:
    """
    Metric: area ratio of the LARGEST connected blob of pixels that sit
    well ABOVE the image's own local ambient-lighting trend, found via
    Gaussian-blur background subtraction (a standard shading-correction
    technique in document image processing).

    text_mask (optional): a uint8 0/255 mask, same H x W as `bgr`,
    marking roughly where printed content (text/lines/logos) actually
    is — see validator.py, which builds this once from a shared Canny
    edge map and reuses it across checks. When provided, any candidate
    glare blob is intersected with this mask before being measured, and
    the area ratio is computed against the size of the text-bearing
    region rather than the whole frame.

    Why: a highlight sitting on a shiny bottle cap, a curved plastic
    shoulder, a foil fold, or anything else in the background/packaging
    that ISN'T where the printed content is has no bearing on whether
    OCR can read the label — flagging it as "glare" only produces false
    rejections of perfectly legible photos. Only a highlight that
    actually overlaps the content region can plausibly erase text, so
    that's the only kind this check should penalize. Without a
    text_mask, the check falls back to its original whole-frame
    behaviour.

    Why not "just find very bright pixels": an ordinary, perfectly valid
    white document page is itself one huge contiguous region of near-
    white pixels. Any check based on an ABSOLUTE brightness cutoff
    inevitably flags most bright documents as "glare", which is
    precisely the mistake this check must avoid.

    Approach:
      1. Estimate the slow-varying local lighting trend by blurring the
         grayscale image with a large sigma — deliberately large enough
         to exceed a typical glare spot's size, so the estimate at a
         highlight's location still reflects the surrounding page rather
         than the highlight itself. For speed, this is approximated by
         downsampling first, blurring the small image, then upsampling
         back (see config.glare_downsample_factor) — a large-kernel blur
         on a full-resolution image is unnecessarily expensive when only
         a smooth low-frequency estimate is needed.
      2. diff = original - blurred (clipped at 0). This is near-zero
         wherever the image matches its own local trend, including
         across an entire uniformly bright page, and large only where a
         genuine local spike (a real reflection) sits above that trend.
      3. Threshold diff, then apply a small morphological opening to
         discard thin/line-shaped fragments (repeated text/ruled lines
         can otherwise chain into one large but spurious "connected
         blob" purely through pattern connectivity — real glare is a
         compact filled region, not a thin line).
      4. Measure the largest remaining connected blob's area ratio.

    Why it matters for OCR: glare locally erases text entirely (pixels
    are saturated to white with no information left to recover), which
    can silently wipe out just the field OCR actually needed, even when
    the rest of the document is perfectly readable.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = gray.shape[:2]

    sigma = max(10.0, min(h, w) * cfg.glare_background_sigma_fraction)

    # Fast approximation of a large-radius blur: downsample, blur small,
    # upsample back (see config.py for why this is safe to do here).
    factor = max(1, cfg.glare_downsample_factor)
    small_w, small_h = max(1, w // factor), max(1, h // factor)
    small = cv2.resize(gray, (small_w, small_h), interpolation=cv2.INTER_AREA)
    small_sigma = max(1.0, sigma / factor)
    blurred_small = cv2.GaussianBlur(small, (0, 0), sigmaX=small_sigma)
    blurred = cv2.resize(blurred_small, (w, h), interpolation=cv2.INTER_LINEAR)

    diff = gray - blurred
    np.clip(diff, 0, None, out=diff)

    glare_mask = (diff > cfg.glare_diff_intensity_threshold).astype(np.uint8)

    if glare_mask.sum() == 0:
        return True, 0.0, ""

    # Strip thin/line-shaped fragments before measuring blob size — see
    # docstring point 3 above.
    open_k = cfg.glare_open_kernel_size
    if open_k > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
        glare_mask = cv2.morphologyEx(glare_mask, cv2.MORPH_OPEN, kernel)

    if glare_mask.sum() == 0:
        return True, 0.0, ""

    # Restrict candidate glare to where the actual content is, if we
    # know that — see docstring above.
    denom_pixels = h * w
    if text_mask is not None:
        glare_mask = cv2.bitwise_and(glare_mask, glare_mask, mask=(text_mask > 0).astype(np.uint8))
        if glare_mask.sum() == 0:
            return True, 0.0, ""
        text_area = int(np.count_nonzero(text_mask))
        if text_area > 0:
            denom_pixels = text_area

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(glare_mask, connectivity=8)
    if num_labels <= 1:
        return True, 0.0, ""

    # stats[0] is the background label; skip it.
    largest_blob_area = int(stats[1:, cv2.CC_STAT_AREA].max())
    largest_blob_ratio = largest_blob_area / float(denom_pixels)

    if largest_blob_ratio > cfg.glare_blob_area_ratio_threshold:
        return False, largest_blob_ratio, "Strong glare or reflection detected on the document."
    return True, largest_blob_ratio, ""


# ---------------------------------------------------------------------
# 7. Blank / near-blank
# ---------------------------------------------------------------------
def check_blank(
    gray: np.ndarray, cfg: ValidationConfig, edges: Optional[np.ndarray] = None
) -> Tuple[bool, float, str]:
    """
    Metric: (a) Canny edge density — fraction of pixels detected as
    edges — and (b) grayscale standard deviation.

    Why BOTH, and not just low std dev alone: a legitimate, clean white
    document (e.g. a mostly-empty form, or a page with just a signature)
    can have low overall intensity variance yet still contains real,
    OCR-relevant edges (the printed text/lines themselves). Requiring
    edge density to ALSO be near-zero before calling an image "blank"
    is what prevents misclassifying a genuinely valid, low-content
    document as blank. An image that is blank in the way this check
    cares about — a photo of an empty desk, a lens cap accidentally
    triggering capture, a mostly uniform surface — has both no texture
    variance AND essentially no edges.

    Why it matters for OCR: there is nothing to recognize, and running
    the OCR pipeline on it wastes downstream compute for a guaranteed
    empty/garbage result.
    """
    std = float(np.std(gray))
    if edges is None:
        edges = cv2.Canny(gray, 50, 150)
    edge_density = float(np.count_nonzero(edges)) / edges.size

    if edge_density < cfg.blank_edge_density_threshold and std < cfg.blank_std_threshold:
        return False, edge_density, "Image appears blank or contains no visible content."
    return True, edge_density, ""


# ---------------------------------------------------------------------
# 8. Relevance / document presence
# ---------------------------------------------------------------------
class DocumentRelevanceChecker:
    """
    Isolated behind its own class so the relevance check can later be
    swapped for a trained lightweight classifier (e.g. a small
    MobileNet-style document/non-document classifier) WITHOUT changing
    validator.py or any other check — only this class's `check()` method
    would need to change.

    Default implementation (lightweight, deterministic, no ML, no
    network calls): document images are dominated by dense, fine-grained
    edge structure (printed text strokes, table/form lines, borders)
    distributed broadly across the frame. Typical "unrelated photo"
    content (a face, a landscape, a single object on a table) tends to
    have much lower overall edge density, or edges concentrated only
    along a small number of large object silhouettes rather than spread
    finely across the frame.

    This is a coarse proxy, not a real document classifier — it will
    have real false positives/negatives (e.g. a heavily patterned fabric
    photo could have high edge density; a document photographed from a
    steep angle with a lot of blank margin could have lower edge density
    than expected). It is intentionally conservative and meant to catch
    the CLEARLY unrelated case cheaply, not to be a strict gate. If
    accuracy needs improve, replace `check()`'s internals with a real
    model call — the interface (input: BGR array, output: bool + score)
    stays the same.
    """

    def __init__(self, cfg: ValidationConfig):
        self.cfg = cfg

    def check(self, gray: np.ndarray, edges: Optional[np.ndarray] = None) -> Tuple[bool, float, str]:
        if edges is None:
            edges = cv2.Canny(gray, 50, 150)
        edge_density = float(np.count_nonzero(edges)) / edges.size
        if edge_density < self.cfg.relevance_min_edge_density:
            return False, edge_density, (
                "Image does not appear to contain a document; "
                "expected content was not detected."
            )
        return True, edge_density, ""
