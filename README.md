# Image Validator — Pre-OCR Image Quality Check

Validates an uploaded packaging/label photo **before** it's sent for OCR.
Checks resolution, blur, brightness, contrast, glare, blank frames, and
whether the image is actually packaging — and returns ACCEPTED/REJECTED
with clear reasons so the frontend can ask the user to re-upload.

## Structure

```
image_validator/
├── main.py                        CLI entry point — validate a file or folder
├── validator.py                   Orchestrates all checks into one result
├── checks.py                      Individual check implementations (blur, glare, etc.)
├── config.py                      All thresholds — tune here, not in check logic
├── models.py                      ValidationResult / CheckMetrics data classes
├── messages.py                    Builds the user-facing rejection message
├── utils.py                       Image loading/decoding helpers
├── relevance_classifier.py        Optional trained-model relevance check (inference)
├── train_relevance_classifier.py  Script to train the above (needs your own dataset)
├── test_validator.py              Automated test suite
├── requirements.txt               Python dependencies
├── photos/                        Sample test images

```

## Requirements

- Python 3.9+
- `pip install -r requirements.txt`

## How to run

Validate one image:
```
python main.py path/to/image.jpg
```

Validate a whole folder, machine-readable output:
```
python main.py path/to/folder --json
```

Run the test suite:
```
python test_validator.py
```

Use as a library (e.g. from another backend script):
```python
from image_validator import ImageValidator

result = ImageValidator().validate("path/to/image.jpg")
if result.valid:
    ...  # forward to OCR
else:
    print(result.reasons)
```

## Note

The packaging-relevance check works out of the box using a lightweight
heuristic — no setup needed. `train_relevance_classifier.py` is an
optional upgrade to a trained model; it needs its own labeled image
dataset (150+ images per class) and is not required to run this module.
