from tabes.samplers.base import BaseSampler, SampleResult
from tabes.samplers.boe import BoEConfig, BoESampler
from tabes.samplers.heuristics import (
    ConfidenceSampler,
    EntropySampler,
    MarginSampler,
    RandomSampler,
)

SAMPLERS = {
    "random": RandomSampler,
    "confidence": ConfidenceSampler,
    "margin": MarginSampler,
    "entropy": EntropySampler,
    "boe": BoESampler,
}

__all__ = [
    "BaseSampler", "SampleResult", "BoEConfig", "BoESampler",
    "ConfidenceSampler", "EntropySampler", "MarginSampler", "RandomSampler",
    "SAMPLERS",
]
