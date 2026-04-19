"""Adaptive Level-Crossing Sampling (ALCS) algorithms."""

def alcs_trigger(signal: list[float], threshold: float) -> list[float]:
    return [s for s in signal if abs(s) > threshold]
