
from __future__ import annotations
import numpy as np

def mean_w(p: float, wAA: float, wAĀ: float, wĀĀ: float) -> float:
    q = 1.0 - p
    return p*p*wAA + 2*p*q*wAĀ + q*q*wĀĀ

def next_p_det(p: float, wAA: float, wAĀ: float, wĀĀ: float) -> float:
    q = 1.0 - p
    wbar = mean_w(p, wAA, wAĀ, wĀĀ)
    if wbar <= 0:
        return p
    return (p*p*wAA + p*q*wAĀ) / wbar

def geno_freqs_after_sel(p: float, wAA: float, wAĀ: float, wĀĀ: float):
    q = 1.0 - p
    wbar = mean_w(p, wAA, wAĀ, wĀĀ)
    if wbar <= 0:
        return p*p, 2*p*q, q*q
    return (p*p*wAA)/wbar, (2*p*q*wAĀ)/wbar, (q*q*wĀĀ)/wbar

def hw_from_p(p: float):
    q = 1.0 - p
    return p*p, 2*p*q, q*q

def next_p_stoch(p: float, wAA: float, wAĀ: float, wĀĀ: float, N: int, rng: np.random.Generator):
    fAA, fAĀ, fĀĀ = geno_freqs_after_sel(p, wAA, wAĀ, wĀĀ)
    counts = rng.multinomial(N, [fAA, fAĀ, fĀĀ])
    p_next = (2*counts[0] + counts[1])/(2.0*N)
    return p_next, counts
