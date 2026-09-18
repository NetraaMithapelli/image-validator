"""
api.py
======

Exposes ImageValidator as a standalone HTTP microservice, so any other
part of the system — the frontend directly, task 1's auth/DB backend,
or task 4's OCR service — can call it over plain HTTP regardless of
what language/framework THEY are built in. This is deliberately the
integration point when you don't yet know (or don't control) the rest
of the stack: an HTTP request/response is the one interface every
backend language can speak, so nobody is blocked waiting on that
decision, and task 3 stays a separately runnable, separately testable
service instead of code someone has to import and wire in by hand.

If it later turns out the main backend IS Python (Flask/FastAPI/Django),
that team can choose to skip the network hop and `from image_validator
import ImageValidator` directly instead — this file doesn't stop that,
it just means you don't have to wait to find out before shipping.

Run standalone:
    python api.py
    (listens on http://0.0.0.0:5000)

Endpoints:
    GET  /health
        -> {"status": "ok"}
        Just confirms the service is up — use this for a Docker/K8s
        health check or a quick "is it running" curl from teammates.

    POST /validate-image
        Multipart form upload, field name "image".
        -> 200 with a JSON body shaped exactly like main.py --json's
           per-image output: {file_path, valid, status, reasons,
           checks, metrics, validation_time_ms}.
        -> 400 if no file was attached, or the field name is wrong.

Example calls (any of these work against the same running service):

    curl -F "image=@dettol.jpg" http://localhost:5000/validate-image

    # Python (e.g. from task 1's backend, or a quick test script)
    import requests
    with open("dettol.jpg", "rb") as f:
        r = requests.post(
            "http://localhost:5000/validate-image",
            files={"image": f},
        )
    print(r.json())

    // Node / Express (if task 1 turns out to be Node)
    const form = new FormData();
    form.append("image", fs.createReadStream("dettol.jpg"));
    const res = await fetch("http://localhost:5000/validate-image", {
        method: "POST",
        body: form,
    });
    const result = await res.json();

Deployment note: this dev server (`python api.py`) is fine for your SIH
demo. For anything beyond that, run it behind a real WSGI server
instead, e.g.:
    pip install gunicorn
    gunicorn -w 2 -b 0.0.0.0:5000 api:app
"""

import os
import sys
import tempfile

from flask import Flask, jsonify, request

# Same sys.path fix main.py uses, so this runs correctly whether you
# launch it as `python api.py` from inside this folder, or
# `python -m image_validator.api` from the project root.
_PACKAGE_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PACKAGE_PARENT not in sys.path:
    sys.path.insert(0, _PACKAGE_PARENT)

from image_validator.config import ValidationConfig
from image_validator.validator import ImageValidator

app = Flask(__name__)

# Built once at process startup (loads the classifier if trained weights
# are present), not per-request — model/config loading is comparatively
# expensive and has no reason to repeat on every upload.
_validator = ImageValidator()

# Matches ValidationConfig.supported_extensions; used only to give a
# clearer error message before even attempting a decode.
_ALLOWED_EXTENSIONS = ValidationConfig().supported_extensions


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/validate-image", methods=["POST"])
def validate_image():
    if "image" not in request.files:
        return jsonify({
            "error": "No file attached under the 'image' form field."
        }), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    suffix = os.path.splitext(file.filename)[1].lower()
    if suffix not in _ALLOWED_EXTENSIONS:
        return jsonify({
            "error": f"Unsupported file type '{suffix}'. "
                     f"Allowed: {', '.join(_ALLOWED_EXTENSIONS)}"
        }), 400

    # Saved to a temp file because ImageValidator.validate() takes a
    # file PATH (so it can work identically for the CLI, this API, and
    # any future caller) rather than raw bytes — this is the one place
    # that bridges "bytes we received over HTTP" to that interface.
    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as f:
            file.save(f)
        result = _validator.validate(tmp_path)
        # Report the caller's original filename, not the temp path,
        # since the temp path is meaningless to whoever called this.
        result.file_path = file.filename
        return jsonify(result.to_dict())
    finally:
        os.remove(tmp_path)


if __name__ == "__main__":
    # host="0.0.0.0" so teammates on the same network/container can
    # reach it too, not just localhost on this machine.
    app.run(host="0.0.0.0", port=5000, debug=True)
