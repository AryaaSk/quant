"""Fractional Kelly with cap.

Kelly: f* = (p*b - q) / b, where b = decimal_odds - 1, q = 1 - p.
We use a fraction (default 25%) of full Kelly and cap each bet at `kelly_cap` of bankroll.
"""
from __future__ import annotations


def kelly_fraction(p_model: float, decimal_odds: float) -> float:
    b = decimal_odds - 1.0
    if b <= 0:
        return 0.0
    q = 1.0 - p_model
    f = (p_model * b - q) / b
    return max(0.0, f)


def position_size(
    p_model: float,
    decimal_odds: float,
    bankroll: float,
    *,
    kelly_fraction_ratio: float = 0.25,
    kelly_cap: float = 0.02,
) -> float:
    f = kelly_fraction(p_model, decimal_odds) * kelly_fraction_ratio
    f = min(f, kelly_cap)
    f = max(0.0, f)
    return bankroll * f
