"""Decision rule: given p_model and p_market, decide BACK / LAY / SKIP."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Decision:
    side: str  # "BACK", "LAY", or "SKIP"
    edge: float
    stake_fraction: float  # of bankroll, clamped


def decide(
    p_model: float,
    p_market: float,
    *,
    commission_bps: int = 20,
    base_slippage_bps: int = 30,
    safety_margin_bps: int = 150,
) -> Decision:
    """Decide direction and signed edge. Position size is left to `kelly.py`."""
    cost = (commission_bps + base_slippage_bps + safety_margin_bps) / 10_000.0
    edge = p_model - p_market
    if abs(edge) <= cost:
        return Decision(side="SKIP", edge=edge, stake_fraction=0.0)
    side = "BACK" if edge > 0 else "LAY"
    return Decision(side=side, edge=edge, stake_fraction=0.0)
