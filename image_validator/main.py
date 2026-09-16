"""
main.py
=======

CLI entry point.

Usage:
    python main.py path/to/image.jpg
    python main.py ./uploaded_images
    python main.py ./uploaded_images --json
"""

import argparse
import json
import os
import sys

# Make `image_validator.*` importable regardless of how this file is
# invoked: `python -m image_validator.main`, `python image_validator/main.py`
# from the project root, or `python main.py` from inside the package
# folder itself. In every case, adding the PARENT of this file's
# directory to sys.path guarantees `import image_validator` resolves,
# since that parent directory is exactly where the `image_validator`
# package folder lives.
_PACKAGE_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PACKAGE_PARENT not in sys.path:
    sys.path.insert(0, _PACKAGE_PARENT)

from image_validator.validator import ImageValidator

# Display-only labels for the checks dict, in a fixed, readable order.
_CHECK_LABELS = [
    ("file_valid", "File valid"),
    ("resolution", "Resolution"),
    ("blur", "Blur"),
    ("brightness", "Brightness"),
    ("contrast", "Contrast"),
    ("glare", "Glare"),
    ("blank", "Blank"),
    ("relevance", "Relevance"),
]


def _print_human(result) -> None:
    print("=" * 40)
    print("IMAGE VALIDATION RESULT")
    print("=" * 40)
    print(f"File: {result.file_path}")
    print()
    print(f"Status: {result.status}")
    print()

    if result.reasons:
        print("Reasons:")
        for r in result.reasons:
            print(f"- {r}")
        print()

    print("Checks:")
    for key, label in _CHECK_LABELS:
        if key not in result.checks:
            continue
        mark = "\u2713" if result.checks[key] else "\u2717"
        print(f"{mark} {label}")
    print()
    print(f"Validation time: {result.validation_time_ms:.1f} ms")
    print()

    if not result.valid:
        print(ImageValidator.user_message(result))
        print()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Pre-OCR image validation (PS 26034)."
    )
    parser.add_argument("path", help="Path to a single image file or a folder of images.")
    parser.add_argument(
        "--json", action="store_true",
        help="Print machine-readable JSON instead of the human-readable report.",
    )
    args = parser.parse_args(argv)

    validator = ImageValidator()
    results = validator.validate_path(args.path)

    if not results:
        print(f"No supported image files found at: {args.path}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps([r.to_dict() for r in results], indent=2))
    else:
        for i, result in enumerate(results):
            if i > 0:
                print()
            _print_human(result)

    # Exit code reflects overall outcome: 0 only if every image passed.
    return 0 if all(r.valid for r in results) else 2


if __name__ == "__main__":
    sys.exit(main())
