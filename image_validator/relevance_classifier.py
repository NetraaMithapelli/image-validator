"""
relevance_classifier.py
========================

Inference-side wrapper around the model trained by
train_relevance_classifier.py. Loads once, then classifies a BGR
(OpenCV-style) image array as "packaging" vs "not packaging" with a
confidence score.

Design choices worth knowing about:

- Import of torch/torchvision is deferred into __init__, not done at
  module load time. This is deliberate: torch is a heavy optional
  dependency, and a deployment that doesn't have it installed (or
  hasn't trained weights yet) should still be able to import this
  module and the rest of image_validator without crashing — it should
  just fall back to the heuristic relevance check in that case. See
  validator.py, which does exactly that: tries to construct this class,
  and on any failure (ImportError, missing weights file, corrupt
  checkpoint) falls back to the existing edge-density-based
  DocumentRelevanceChecker instead of raising.
- is_available() lets validator.py check upfront rather than relying
  purely on exception handling at prediction time.
"""

from typing import Optional, Tuple

import numpy as np


class PackagingRelevanceClassifier:
    """
    Usage:
        clf = PackagingRelevanceClassifier("relevance_classifier.pt")
        if clf.is_available():
            is_relevant, confidence = clf.predict(bgr_image)
    """

    IMAGE_SIZE = 224

    def __init__(self, weights_path: str, device: Optional[str] = None):
        self._model = None
        self._transform = None
        self._device = None
        self._load_error: Optional[str] = None

        try:
            import torch
            from torchvision import models, transforms

            self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")

            model = models.mobilenet_v3_small(weights=None)
            in_features = model.classifier[-1].in_features
            model.classifier[-1] = torch.nn.Linear(in_features, 2)
            state_dict = torch.load(weights_path, map_location=self._device)
            model.load_state_dict(state_dict)
            model.eval()
            model.to(self._device)

            self._model = model
            self._torch = torch
            self._transform = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.Resize(256),
                    transforms.CenterCrop(self.IMAGE_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
        except Exception as exc:  # noqa: BLE001 - deliberately broad, see docstring
            self._load_error = f"{type(exc).__name__}: {exc}"

    def is_available(self) -> bool:
        return self._model is not None

    def load_error(self) -> Optional[str]:
        """Why loading failed, if it did — useful for a one-time startup log line."""
        return self._load_error

    def predict(self, bgr_image: np.ndarray) -> Tuple[bool, float]:
        """
        Returns (is_packaging, confidence) where confidence is the
        model's softmax probability for the predicted class, in [0, 1].
        Raises RuntimeError if the model failed to load — callers should
        check is_available() first (validator.py does).
        """
        if not self.is_available():
            raise RuntimeError(f"Classifier not loaded: {self._load_error}")

        rgb = bgr_image[:, :, ::-1].copy()  # BGR (OpenCV) -> RGB (torchvision expects this)
        tensor = self._transform(rgb).unsqueeze(0).to(self._device)

        with self._torch.no_grad():
            logits = self._model(tensor)
            probs = self._torch.softmax(logits, dim=1)[0]
            pred_class = int(probs.argmax().item())
            confidence = float(probs[pred_class].item())

        is_packaging = pred_class == 1  # class 1 == "packaging", see CLASS_NAMES in training script
        return is_packaging, confidence
