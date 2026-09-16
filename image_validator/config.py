"""
config.py
=========

Every threshold used anywhere in the validator lives here, in one place,
so the whole system can be tuned without touching check logic.

IMPORTANT — read before changing numbers
-----------------------------------------
Some of these thresholds correspond to well-known, widely used heuristics
in the OCR / document-image-processing community (e.g. variance-of-
Laplacian for blur, mean-intensity for exposure). They are NOT formal
international standards (there is no ISO spec for "how blurry is too
blurry"). They are commonly cited *starting points* from CV practice and
OCR-preprocessing literature (OpenCV docs, Tesseract's own image-quality
guidance, and widely replicated blog/paper heuristics such as the
"variance of Laplacian < 100 => blurry" rule popularized by
Pyimagesearch and used in many production OCR pre-checks).

None of them should be treated as final. Section 8 of the README explains
how to calibrate every value below against your own labelled
valid/invalid dataset (ROC / precision-recall sweep). Treat these as
"reasonable defaults that will not embarrass you on day one", not as
ground truth.
"""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class ValidationConfig:
    # ------------------------------------------------------------------
    # General / I/O
    # ------------------------------------------------------------------
    # Formats we accept. OCR engines (Tesseract, cloud OCR APIs) all
    # support these three natively; anything else is converted upstream
    # or rejected here rather than risking a silent bad decode.
    supported_extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp")

    # All quality checks (blur/brightness/contrast/glare/blank/relevance)
    # run on a downscaled copy of the image. Full-resolution analysis of
    # a 12MP phone photo is wasted work for a variance/mean/std computation
    # that converges statistically long before that. This keeps the
    # module low-latency regardless of the uploaded image's size.
    # 1000px on the long edge is generous for statistical texture/exposure
    # measures while staying fast (OpenCV resize is O(pixels), so this is
    # a large, predictable latency win on modern phone-camera uploads).
    analysis_max_dimension: int = 1000

    # ------------------------------------------------------------------
    # 1. File validity
    # ------------------------------------------------------------------
    # No threshold here — either OpenCV/Pillow can decode the bytes into
    # a valid image array or it can't.

    # ------------------------------------------------------------------
    # 2. Resolution
    # ------------------------------------------------------------------
    # There is no single universal "minimum OCR resolution" — accuracy
    # depends on DPI at *print* size, not raw pixel count. But absent DPI
    # metadata (most phone/camera uploads don't carry meaningful DPI),
    # pixel dimensions are the only practical proxy we have at upload time.
    #
    # Reference points used to pick a starting value:
    #   - Tesseract's own documentation recommends ~300 DPI for a scanned
    #     page, and states recognition degrades noticeably below ~150 DPI
    #     equivalent character height.
    #   - A commonly used rule of thumb in OCR preprocessing is that
    #     individual character x-height should be at least ~20-30px for
    #     reliable segmentation.
    #   - VGA resolution (640x480 = ~300K px) is a widely used floor in
    #     document-capture SDKs (e.g. many mobile document scanners reject
    #     below this) as "unlikely to contain legible text at normal
    #     document-to-camera distance".
    #
    # We use 640x480 as the floor. This is a heuristic proxy, not a DPI
    # guarantee — a 640x480 image of a tiny receipt held far from the
    # camera can still fail OCR, and a 700x700 crop of a large-font sign
    # can succeed. Calibrate against your dataset (see README).
    min_width_px: int = 640
    min_height_px: int = 480

    # ------------------------------------------------------------------
    # 3. Blur (sharpness)
    # ------------------------------------------------------------------
    # Metric: variance of the Laplacian (second derivative / edge energy).
    # A sharp image has many strong edges -> high variance. A blurred
    # image smooths edges out -> low variance. This is one of the most
    # widely used lightweight blur metrics in applied CV (no ML, one
    # convolution, O(pixels)).
    #
    # Threshold: 100.0 is a widely *cited* starting point in OCR/document
    # pre-processing pipelines and blur-detection tutorials, but it is a
    # community heuristic, not a formal standard — the raw score is scale-
    # and content-dependent (a photo of a mostly-blank form has fewer
    # edges than a dense paragraph even both perfectly in focus). This is
    # exactly the kind of value that MUST be recalibrated on real upload
    # data (see README section on calibration).
    blur_laplacian_threshold: float = 100.0

    # ------------------------------------------------------------------
    # 4. Brightness / exposure
    # ------------------------------------------------------------------
    # Metric: mean pixel intensity of the grayscale image, 0-255.
    # Below min -> underexposed/too dark to read reliably.
    # Above max -> overexposed/washed out (detail clipped to white).
    #
    # There is no universal standard for "too dark" here either. 50 and
    # 200 (out of 255) are commonly used conservative bounds in exposure-
    # QA heuristics — they flag only clearly problematic images, leaving
    # normally-lit documents (which usually sit in the 100-180 range)
    # untouched.
    brightness_min: float = 50.0
    # Raised from a stricter 200: real product-label photos legitimately
    # sit well above 200 whenever the packaging/backdrop is light-toned
    # (pastel bottles, white studio backgrounds) even when the text
    # itself has excellent local contrast and is perfectly legible —
    # verified against a shampoo bottle sample that measured 220.7 and
    # is completely readable. Genuine overexposure (text actually
    # washed out) shows up as high brightness AND low contrast together,
    # which contrast_std_min below already catches independently.
    brightness_max: float = 235.0

    # Pixels at or below trim_low / at or above trim_high are excluded
    # before computing brightness/contrast (see checks._trimmed_values).
    # These aren't "background" pixels specifically — we deliberately
    # don't assume anything about where or what the background is, since
    # real uploads (unlike studio product shots) can have any background
    # at all. They're simply the fully-saturated pixels that carry no
    # tonal information either way, on ANY background.
    trim_low: int = 5
    trim_high: int = 250

    # ------------------------------------------------------------------
    # 5. Contrast
    # ------------------------------------------------------------------
    # Metric: standard deviation of grayscale pixel intensities.
    # A document with visible dark text on a light background has a wide
    # spread of intensities -> higher std dev. A washed-out, low-contrast
    # scan clusters tightly around one value -> low std dev.
    #
    # 30.0 is a commonly used low-contrast heuristic floor in image-QA
    # code (not a formal standard). It intentionally sits low so that
    # normal, slightly flat-lit documents are not rejected — this check
    # is meant to catch clearly washed-out captures, not to enforce
    # "punchy" contrast.
    contrast_std_min: float = 30.0

    # ------------------------------------------------------------------
    # 6. Glare / reflection
    # ------------------------------------------------------------------
    # A naive "pixel is very bright" test cannot distinguish real glare
    # from a perfectly ordinary bright white document page — a plain
    # white background is, by definition, one giant contiguous region of
    # near-255 pixels, so any absolute-brightness threshold flags most
    # normal documents as "glare".
    #
    # Approach actually used: illumination/background subtraction via a
    # heavily-blurred copy of the image — a standard shading-correction
    # technique in document image processing.
    #   1. blurred = the grayscale image with a very large Gaussian blur
    #      applied. This approximates the slow-varying ambient-lighting
    #      trend of the scene while washing out anything much smaller
    #      than the blur radius (ordinary text, fine texture).
    #   2. diff = original - blurred, clipped to >= 0. Where the image
    #      matches its own local lighting trend — including across an
    #      entire uniformly-lit bright page — diff is close to zero.
    #      Only a genuine LOCAL spike above the ambient trend (a real
    #      specular highlight/reflection) produces a large positive diff.
    #   3. A small morphological opening is applied to the resulting
    #      binary mask before measuring blob size, to discard thin,
    #      line-like fragments (e.g. a bright ruled line, or a repeated
    #      text pattern) that can otherwise chain together into a
    #      large-but-spurious "connected blob" through pattern
    #      connectivity alone — real glare is a compact, filled region,
    #      not a thin connected line.
    #
    # Why the blur radius must be genuinely large (not just "bigger than
    # a text stroke"): if the blur radius is smaller than or comparable
    # to the glare spot itself, the blur mostly averages WITHIN the
    # bright spot, so the local "background" estimate at the spot's
    # center gets pulled up toward the glare's own brightness — erasing
    # the very excess we are trying to detect. The radius must clearly
    # exceed a typical glare spot's size so the local-background estimate
    # at its center still reflects the surrounding page. A quarter of the
    # shorter image dimension is a practical starting point for "clearly
    # larger than a compact reflection, much smaller than the whole
    # page" — a heuristic, not a derived constant, and it was arrived at
    # empirically against synthetic bright-document and localized-glare
    # test images during development, not from a citable standard.
    glare_background_sigma_fraction: float = 0.25

    # Directly running a Gaussian blur with a sigma this large on a
    # full-resolution analysis image is needlessly expensive (a wide
    # kernel is a lot of arithmetic per pixel). Since we only need a
    # smooth, low-frequency estimate of the lighting trend — not
    # per-pixel precision — we downsample by this factor first, blur
    # the much smaller image (with sigma scaled down by the same
    # factor), then upsample the result back. This is a standard,
    # numerically very close approximation of a large-radius blur, and
    # empirically ran ~100x+ faster in this module (335ms -> ~3ms on a
    # ~1000px test image) with an identical glare-detection outcome on
    # every test case checked during development. This is purely a
    # performance optimization, not a detection-accuracy trade-off.
    glare_downsample_factor: int = 8

    # Minimum excess (0-255 scale) over the local lighting trend for a
    # pixel to count as a highlight candidate. Genuine specular glare
    # reads roughly 30-50+ levels above its surrounding page; ordinary
    # gentle lighting unevenness (a window on one side of the shot, mild
    # vignetting) is a much slower, gentler gradient than this.
    glare_diff_intensity_threshold: int = 40

    # Kernel size for the pre-measurement morphological opening that
    # strips thin/line-shaped mask fragments before blob size is
    # measured (see step 3 above). Small and fixed — this is about
    # removing hairline artifacts, not scaling with page size.
    glare_open_kernel_size: int = 9

    # A single remaining connected highlight blob covering more than
    # this fraction of the content region (see check_glare's text_mask)
    # is flagged as glare.
    #
    # IMPORTANT — this was deliberately set loose, not tight, based on
    # measured data, not guessed: across real product-label photos, the
    # ratio does NOT cleanly separate "legible" from "damaged":
    #   kitkat_correct.png (fully legible, real)........... 0.017
    #   synthetic glare fully covering real text (bad)...... 0.025
    #   dettol.jpg  (fully legible, real — printed white
    #                barcode/callout boxes, not reflections). 0.060
    #   marie.jpg   (fully legible, real — genuine specular
    #                reflection, but on a plain foil area with
    #                no text under it)....................... 0.075
    #   dettol.webp (same packaging as dettol.jpg)........... 0.115
    #   invisible_text.jpg (genuinely unreadable input)...... 0.157
    # A pixel-brightness heuristic cannot distinguish "printed white ink"
    # from "camera reflection" — both hit near-255 in the photo — so
    # legible real photos and a damaging synthetic case land in
    # overlapping ranges. No single threshold here can be both strict
    # and safe. This value is set high enough to only catch clearly
    # extreme cases (comparable to invisible_text.jpg) and deliberately
    # avoids rejecting legible real photos; treat glare_ratio as a
    # logged/advisory metric rather than a precise pass/fail signal, and
    # let the OCR + rule-engine stages downstream be the authoritative
    # check for whether a specific mandatory field was actually readable.
    # Revisit only with a proper labelled dataset, or by folding this
    # into the same trained classifier as relevance (see
    # relevance_classifier.py) rather than more brightness heuristics.
    glare_blob_area_ratio_threshold: float = 0.13

    # ------------------------------------------------------------------
    # 7. Blank / near-blank detection
    # ------------------------------------------------------------------
    # A truly blank/near-blank image has almost no edges (nothing was
    # written/printed, or the camera is pointed at a flat empty surface)
    # AND very low intensity variance (nothing to see at all).
    #
    # We deliberately require *both* low edge density AND low contrast to
    # fire, specifically so that a legitimate bright/white document (which
    # has low background variance but plenty of edges from printed text)
    # is not misclassified as blank.
    blank_edge_density_threshold: float = 0.001  # fraction of pixels that are Canny edges
    blank_std_threshold: float = 8.0

    # ------------------------------------------------------------------
    # 8. Relevance / document-presence
    # ------------------------------------------------------------------
    # Lightweight, non-ML heuristic: documents are dominated by strong,
    # regularly distributed edges (text strokes, table lines, borders)
    # concentrated in structured regions, unlike most "unrelated photo"
    # content (faces, landscapes, single large objects) which tends to
    # have far lower edge density or edges concentrated only on object
    # silhouettes. This is a coarse proxy, explicitly isolated behind
    # DocumentRelevanceChecker (see checks.py) so it can be swapped for a
    # trained lightweight classifier later without touching the rest of
    # the pipeline.
    relevance_min_edge_density: float = 0.02

    # Optional upgrade path: a trained MobileNetV3 binary classifier
    # (see relevance_classifier.py, train_relevance_classifier.py) that
    # replaces the heuristic above once you've trained it on real
    # packaging vs. non-packaging examples. Set to None (or leave the
    # file missing) to run purely on the heuristic — nothing else in this
    # module requires torch to be installed.
    relevance_classifier_weights_path: str = "relevance_classifier.pt"

    # A negative ("not packaging") classifier prediction only rejects the
    # image when the model's confidence exceeds this — an uncertain call
    # fails OPEN (image is accepted, left to downstream OCR/rule-engine
    # stages) rather than closed. Same reasoning as glare_blob_area_ratio_
    # threshold above: an early-stage filter with a small training set
    # should be conservative about hard-rejecting on a shaky signal.
    relevance_classifier_min_confidence: float = 0.65

    # ------------------------------------------------------------------
    # Performance
    # ------------------------------------------------------------------
    # Individual checks can be toggled off without touching orchestration
    # logic — useful for A/B-ing which checks help vs. hurt once you have
    # real acceptance/rejection data.
    enabled_checks: dict = field(default_factory=lambda: {
        "file_valid": True,
        "resolution": True,
        "blur": True,
        "brightness": True,
        "contrast": True,
        "glare": True,
        "blank": True,
        "relevance": True,
    })


DEFAULT_CONFIG = ValidationConfig()
