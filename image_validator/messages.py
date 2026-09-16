"""
messages.py
===========

Converts a ValidationResult's technical `checks` dict into a friendly,
non-technical message for end users — no "Laplacian", no "HSV", no
"variance". Kept separate from validator.py/checks.py so the technical
side can change its internals freely without ever risking a confusing
term leaking into user-facing copy.
"""

from image_validator.models import ValidationResult

# One user-facing bullet + one actionable tip per check, keyed by the
# same check names used in ValidationResult.checks / config.enabled_checks.
_USER_MESSAGES = {
    "file_valid": (
        "The file could not be read as an image.",
        "Please upload a valid JPG, PNG, or WEBP file.",
    ),
    "resolution": (
        "The image resolution is too low.",
        "Move closer to the document or use a higher-resolution camera setting.",
    ),
    "blur": (
        "The image is too blurry.",
        "Keep the camera steady and let it focus before capturing.",
    ),
    "brightness": (
        "The lighting is too dark or too bright.",
        "Use even, sufficient lighting and avoid direct flash.",
    ),
    "contrast": (
        "The document text is not clearly visible.",
        "Make sure there's good lighting contrast between the text and background.",
    ),
    "glare": (
        "Glare is covering part of the document.",
        "Tilt the document or camera slightly to avoid reflections.",
    ),
    "blank": (
        "The image appears to be blank.",
        "Make sure the document is actually in frame before capturing.",
    ),
    "relevance": (
        "The expected document was not detected in the image.",
        "Make sure the complete document is visible and fills most of the frame.",
    ),
}

_GENERIC_TIPS = [
    "Keep the camera steady.",
    "Use sufficient, even lighting.",
    "Avoid reflections on the document.",
    "Make sure the complete document is visible.",
]


def build_user_message(result: ValidationResult) -> str:
    """
    Build the final, user-facing string for a REJECTED result. ACCEPTED
    results don't need this — the caller simply proceeds to OCR.
    """
    if result.valid:
        return "Image accepted."

    failed_checks = [name for name, passed in result.checks.items() if not passed]

    bullets = []
    tips = []
    for name in failed_checks:
        msg, tip = _USER_MESSAGES.get(name, ("The image failed a quality check.", None))
        bullets.append(f"\u2022 {msg}")
        if tip and tip not in tips:
            tips.append(tip)

    if not tips:
        tips = _GENERIC_TIPS

    lines = [
        "Image rejected.",
        "",
        "Please upload a clearer image because:",
        *bullets,
        "",
        "Tips:",
        *[f"\u2022 {t}" for t in tips],
    ]
    return "\n".join(lines)
