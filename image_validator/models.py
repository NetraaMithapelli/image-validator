"""
models.py
=========

Structured, typed containers for validation output. Using dataclasses
(instead of raw dicts everywhere) gives IDE autocomplete and catches typos
in field names early, while `to_dict()` still gives callers/JSON APIs the
plain-dict shape they usually want.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional


@dataclass
class CheckMetrics:
    """Raw numeric outputs from each check, for logging/calibration/debugging."""
    width: Optional[int] = None
    height: Optional[int] = None
    blur_score: Optional[float] = None
    brightness_score: Optional[float] = None
    contrast_score: Optional[float] = None
    glare_ratio: Optional[float] = None
    edge_density: Optional[float] = None

    def to_dict(self) -> Dict:
        # Drop keys that were never computed (e.g. because the check was
        # disabled, or an earlier check already failed and we short-circuited)
        # so the output stays clean rather than full of nulls.
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class ValidationResult:
    """
    The full, structured outcome of validating a single image.

    `valid` / `status` are the decision. `reasons` are short, still-
    technical-ish strings (see messages.py for the fully user-facing
    version). `checks` records True/False per individual check so a
    caller can see *which* check(s) failed. `metrics` carries the raw
    numbers for logging and future threshold calibration.
    """
    file_path: str
    valid: bool
    status: str  # "ACCEPTED" or "REJECTED"
    reasons: List[str] = field(default_factory=list)
    checks: Dict[str, bool] = field(default_factory=dict)
    metrics: CheckMetrics = field(default_factory=CheckMetrics)
    validation_time_ms: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "file_path": self.file_path,
            "valid": self.valid,
            "status": self.status,
            "reasons": self.reasons,
            "checks": self.checks,
            "metrics": self.metrics.to_dict(),
            "validation_time_ms": round(self.validation_time_ms, 2),
        }
