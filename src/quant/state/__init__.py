"""State vector block builders.

A state vector is a fixed-size numeric representation of everything we know at a timestep.
Per-market yaml composes blocks (numeric, market_state, text, calendar) into a contiguous
vector. See ARCHITECTURE.md for design rationale.
"""

from quant.state.composer import StateBuilder, StateVectorSpec  # noqa: F401
