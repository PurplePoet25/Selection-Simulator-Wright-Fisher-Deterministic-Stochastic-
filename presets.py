
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Literal

PRESETS = [
    "A favored — additive",
    "A favored — recessive",
    "A favored — dominant",
    "Overdominance (AĀ advantage)",
    "Underdominance (AĀ disadvantage)",
    "Custom (s,h) / (w's)",
]

# Defaults for fitness dominance (h_v) used by directional presets
DEFAULT_H = {
    "A favored — additive": 0.5,
    "A favored — recessive": 0.0,
    "A favored — dominant": 1.0,
}

CustomMode = Literal["sh", "w"]

@dataclass
class FitnessSpec:
    wAA: float
    wAĀ: float
    wĀĀ: float

def compute_w(preset: str, s: float, h_v: float, custom_ws: Tuple[float,float,float], custom_mode: CustomMode) -> FitnessSpec:
    """Return fitness triplet used by the engine."""
    if preset == "Overdominance (AĀ advantage)":
        return FitnessSpec(0.90, 1.00, 0.80)
    if preset == "Underdominance (AĀ disadvantage)":
        return FitnessSpec(1.00, 0.90, 1.00)
    if preset == "Custom (s,h) / (w's)":
        if custom_mode == "w":
            wAA, wAĀ, wĀĀ = custom_ws
            return FitnessSpec(float(wAA), float(wAĀ), float(wĀĀ))
        else:
            return FitnessSpec(1.0 + s, 1.0 + h_v*s, 1.0)
    # Directional presets use s, h_v live (defaults are set when the preset is chosen)
    return FitnessSpec(1.0 + s, 1.0 + h_v*s, 1.0)

def default_h_for_preset(preset: str) -> float | None:
    return DEFAULT_H.get(preset, None)
