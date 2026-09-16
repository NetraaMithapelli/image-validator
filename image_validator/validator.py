"""
validator.py
============

Orchestrates all individual checks (checks.py) into one structured
ValidationResult per image, and provides folder-level batch validation.

Design notes:
- The image is decoded and downscaled exactly ONCE per validation call.
  Every quality check (blur/brightness/contrast/glare/blank/relevance)
  operates on the same downscaled analysis image and, where applicable,
  the same grayscale conversion of it, so we never redundantly re-read
  or re-convert the image. Resolution is the one check that intentionally
  looks at the ORIGINAL (non-downscaled) dimensions.
- If the file can't even be decoded, we short-circuit immediately — every
  other check needs a valid pixel array, so there is nothing further to
  measure and no point spending latency trying.
- All other checks always run, even if an earlier one already failed,
  so the caller gets a COMPLETE list of everything wrong with the image
  in one pass (matches the spec's example showing multiple simultaneous
  reasons) rather than one-reason-at-a-time round trips.
"""

import os
from typing import List

import cv2
import numpy as np

from image_validator.config import ValidationConfig, DEFAULT_CONFIG
from image_validator.models import ValidationResult, CheckMetrics
from image_validator.checks import (
    check_file_validity,
    check_resolution,
    check_blur,
    check_brightness,
    check_contrast,
    check_glare,
    check_blank,
    DocumentRelevanceChecker,
)
from image_validator.messages import build_user_message
from image_validator.relevance_classifier import PackagingRelevanceClassifier
from image_validator.utils import safe_imread, downscale_for_analysis, to_gray, timer, discover_images
import sys


class ImageValidator:
    """
    Main entry point. Usage:

        validator = ImageValidator()
        result = validator.validate("uploaded_images/image1.jpg")
        if result.valid:
            ... hand off to OCR ...
        else:
            print(validator.user_message(result))
    """

    def __init__(self, config: ValidationConfig = None):
        self.cfg = config or DEFAULT_CONFIG
        # Always-available fallback — zero dependencies, zero setup.
        self._relevance_checker = DocumentRelevanceChecker(self.cfg)

        # Optional upgrade: a trained MobileNetV3 packaging/not-packaging
        # classifier (see relevance_classifier.py + train_relevance_
        # classifier.py). This is genuinely optional — if torch isn't
        # installed, or the weights file doesn't exist yet (no one has
        # run the training script), PackagingRelevanceClassifier catches
        # that internally and reports is_available()=False, and we
        # silently keep using the heuristic checker above. Nothing here
        # requires torch to be present for the rest of the module to work.
        weights_path = self.cfg.relevance_classifier_weights_path
        self._relevance_classifier = None
        if weights_path:
            if not os.path.isabs(weights_path):
                weights_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), weights_path)
            clf = PackagingRelevanceClassifier(weights_path)
            if clf.is_available():
                self._relevance_classifier = clf
            else:
                # One-time, non-fatal notice — not raised, since running
                # without the classifier is a fully supported mode.
                # Printed to stderr, not stdout, so it never corrupts
                # --json output for callers piping/parsing it.
                print(
                    f"[image_validator] Packaging relevance classifier not loaded "
                    f"({clf.load_error()}); using the heuristic edge-density "
                    f"relevance check instead. Run train_relevance_classifier.py "
                    f"to enable the classifier.",
                    file=sys.stderr,
                )

    # ------------------------------------------------------------------
    def validate(self, file_path: str) -> ValidationResult:
        """
        Validate a single image file and return a full ValidationResult.
        Never raises — any failure mode (missing file, corrupt image,
        unreadable format) is captured as a REJECTED result instead.
        """
        with timer() as t:
            result = self._validate_inner(file_path)
        result.validation_time_ms = t.elapsed_ms
        return result

    def _validate_inner(self, file_path: str) -> ValidationResult:
        checks = {}
        reasons: List[str] = []
        metrics = CheckMetrics()
        enabled = self.cfg.enabled_checks

        img = safe_imread(file_path)
        file_ok, file_reason = check_file_validity(img)
        checks["file_valid"] = file_ok
        if not file_ok:
            reasons.append(file_reason)
            # Nothing further can be measured without a decoded image.
            return ValidationResult(
                file_path=file_path,
                valid=False,
                status="REJECTED",
                reasons=reasons,
                checks=checks,
                metrics=metrics,
            )

        # --- Resolution: measured on the ORIGINAL image ---
        if enabled.get("resolution", True):
            res_ok, w, h, res_reason = check_resolution(img, self.cfg)
            checks["resolution"] = res_ok
            metrics.width, metrics.height = w, h
            if not res_ok:
                reasons.append(res_reason)
        else:
            metrics.width, metrics.height = img.shape[1], img.shape[0]

        # --- Everything below shares one downscaled analysis image ---
        analysis_img = downscale_for_analysis(img, self.cfg.analysis_max_dimension)
        gray = to_gray(analysis_img)

        # One shared Canny edge map, reused by blank/relevance (as
        # before) and to derive glare's text_mask (new) — avoids
        # recomputing Canny three times per image.
        edges = cv2.Canny(gray, 50, 150)
        text_mask = self._build_content_region_mask(edges)

        if enabled.get("blur", True):
            ok, score, reason = check_blur(gray, self.cfg)
            checks["blur"] = ok
            metrics.blur_score = score
            if not ok:
                reasons.append(reason)

        if enabled.get("brightness", True):
            ok, score, reason = check_brightness(gray, self.cfg)
            checks["brightness"] = ok
            metrics.brightness_score = score
            if not ok:
                reasons.append(reason)

        if enabled.get("contrast", True):
            ok, score, reason = check_contrast(gray, self.cfg)
            checks["contrast"] = ok
            metrics.contrast_score = score
            if not ok:
                reasons.append(reason)

        if enabled.get("glare", True):
            ok, score, reason = check_glare(analysis_img, self.cfg, text_mask=text_mask)
            checks["glare"] = ok
            metrics.glare_ratio = score
            if not ok:
                reasons.append(reason)

        if enabled.get("blank", True):
            ok, score, reason = check_blank(gray, self.cfg, edges=edges)
            checks["blank"] = ok
            metrics.edge_density = score
            if not ok:
                reasons.append(reason)

        if enabled.get("relevance", True):
            if self._relevance_classifier is not None:
                try:
                    is_packaging, confidence = self._relevance_classifier.predict(analysis_img)
                    # Only let a NEGATIVE prediction reject when the model is
                    # reasonably confident about it — an uncertain call
                    # should fail open (accept, let downstream OCR/rule
                    # stages be the final word) rather than fail closed,
                    # consistent with how glare is treated elsewhere here.
                    ok = is_packaging or confidence < self.cfg.relevance_classifier_min_confidence
                    score = confidence
                    reason = "" if ok else (
                        "Image does not appear to be a photo of packaging or a "
                        "product label."
                    )
                except Exception as exc:  # noqa: BLE001 - degrade, don't fail the request
                    ok, score, reason = self._relevance_checker.check(gray, edges=edges)
            else:
                ok, score, reason = self._relevance_checker.check(gray, edges=edges)
            checks["relevance"] = ok
            # Reuse edge_density slot only if blank check didn't already
            # set it more informatively; both derive from the same Canny
            # pass conceptually but relevance uses its own threshold.
            if metrics.edge_density is None:
                metrics.edge_density = score
            if not ok:
                reasons.append(reason)

        valid = all(checks.values())
        return ValidationResult(
            file_path=file_path,
            valid=valid,
            status="ACCEPTED" if valid else "REJECTED",
            reasons=reasons,
            checks=checks,
            metrics=metrics,
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _build_content_region_mask(edges: np.ndarray) -> np.ndarray:
        """
        Build a mask of solid, filled BLOCKS covering wherever printed
        content (text/logos/tables/barcodes) roughly lives, for
        check_glare's text_mask.

        Deliberately NOT just "dilate the edge map a bit": a thin
        dilation only marks pixels immediately around edges that are
        CURRENTLY visible. Real glare that fully saturates part of a
        label erases exactly the edges that would have been there —
        that's the whole reason it damages OCR — so a thin edge-dilation
        mask would systematically exclude the worst glare from being
        counted at all (verified against a synthetic glare-over-text
        test case during development).

        Instead: dilate+close first to bridge small gaps between
        characters/lines into solid blobs, then take each surviving
        blob's full bounding rectangle as solid mask. A glare spot that
        sits INSIDE a content block's bounding box still counts even if
        it locally has zero edges (because the surrounding text in that
        same block does), while a highlight sitting well outside any
        content block (a bottle cap, a blank margin, a separate part of
        the packaging with no text nearby) is correctly excluded.
        """
        h, w = edges.shape[:2]
        bridged = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
        bridged = cv2.morphologyEx(
            bridged, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
        )

        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(bridged, connectivity=8)
        mask = np.zeros((h, w), dtype=np.uint8)
        if num_labels <= 1:
            return mask  # no structure found at all; nothing to protect

        # Ignore tiny speckle components (noise, not real content).
        min_area = 0.001 * h * w
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < min_area:
                continue
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            bw = stats[i, cv2.CC_STAT_WIDTH]
            bh = stats[i, cv2.CC_STAT_HEIGHT]
            mask[y:y + bh, x:x + bw] = 255
        return mask

    # ------------------------------------------------------------------
    def validate_path(self, path: str) -> List[ValidationResult]:
        """
        Accepts either a single image path or a folder of images.
        Returns a list of ValidationResult (length 1 for a single file).
        """
        paths = discover_images(path, self.cfg)
        return [self.validate(p) for p in paths]

    # ------------------------------------------------------------------
    @staticmethod
    def user_message(result: ValidationResult) -> str:
        """Convenience wrapper so callers don't need to import messages.py directly."""
        return build_user_message(result)
