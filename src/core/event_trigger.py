"""Adaptive Level-Crossing Sampling (ALCS) algorithms."""

def alcs_trigger(signal, threshold):
    return [s for s in signal if abs(s) > threshold]
