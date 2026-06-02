"""Stereo disparity method names for OpenCV classical stereo."""

from __future__ import annotations

DEFAULT_STEREO_METHOD = "stereobm"

STEREO_METHOD_ALIASES = {
    "carpet": "stereobm",  # legacy CLI name
    "bm": "stereobm",
}

VALID_STEREO_METHODS = frozenset({"sgbm", "stereobm", "flow", "blend"})


def normalize_stereo_method(method: str) -> str:
    key = method.strip().lower()
    key = STEREO_METHOD_ALIASES.get(key, key)
    if key not in VALID_STEREO_METHODS:
        raise ValueError(
            f"Unknown stereo method {method!r}. "
            f"Choose: {sorted(VALID_STEREO_METHODS)} (aliases: carpet, bm → stereobm)"
        )
    return key
