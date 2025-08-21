import numpy as np
from dataclasses import dataclass

@dataclass
class BetaPosteriorResult:
    prob_B_wins: float
    median_lift: float | None
    p05: float | None
    p95: float | None

def beta_binomial_compare(sA: int, nA: int, sB: int, nB: int, draws: int = 200_000) -> BetaPosteriorResult:
    # Beta(1,1) priors (uniform)
    aA, bA = 1 + sA, 1 + (nA - sA)
    aB, bB = 1 + sB, 1 + (nB - sB)
    SA = np.random.beta(aA, bA, draws)
    SB = np.random.beta(aB, bB, draws)
    prob = float((SB > SA).mean())
    # Avoid division by zero; lift as (SB - SA) / SA
    safe = SA > 0
    lift = np.where(safe, (SB - SA) / SA, np.nan)
    lift = lift[np.isfinite(lift)]
    if lift.size == 0:
        return BetaPosteriorResult(prob, None, None, None)
    return BetaPosteriorResult(
        prob_B_wins=prob,
        median_lift=float(np.nanmedian(lift)),
        p05=float(np.nanpercentile(lift, 5)),
        p95=float(np.nanpercentile(lift, 95))
    )
