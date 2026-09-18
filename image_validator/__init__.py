"""
image_validator
================
Pre-OCR image quality gate for uploaded packaging/label photos:
resolution, blur, brightness, contrast, glare, blank-frame, and
packaging-relevance checks, combined into one ACCEPTED/REJECTED
decision with human-readable reasons.

This file exists purely so `from image_validator import ImageValidator`
works (needed by test_validator.py and any external caller importing
the package by name) — without it, Python can still import individual
submodules (image_validator.validator, etc.) via implicit namespace
packages, but not the package-level re-export.
"""

from image_validator.validator import ImageValidator
from image_validator.config import ValidationConfig, DEFAULT_CONFIG
from image_validator.models import ValidationResult, CheckMetrics

__all__ = [
    "ImageValidator",
    "ValidationConfig",
    "DEFAULT_CONFIG",
    "ValidationResult",
    "CheckMetrics",
]
