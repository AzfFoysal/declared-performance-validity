"""Declared-performance validity monitoring.

Two-tier design:
  Tier 1 (label-free early warning): an accuracy estimate from unlabeled deployment
          data (Average Thresholded Confidence, Garg et al. 2022). It can only change
          HOW OFTEN we audit. It can never invalidate a declaration on its own.
  Tier 2 (confirmatory): a sequential "testing by betting" monitor on audited labels
          (Timans et al. 2025; Waudby-Smith & Ramdas 2024). Tests
              H0: E[loss_t | past] <= eps_star  for all t
          and rejects when the wealth process reaches 1/delta. By Ville's inequality the
          probability of ever rejecting while H0 holds is at most delta.

Why Tier 1 does not break Tier 2's guarantee:
  Tier 1 only sets the audit RATE for the next window, using past data. WHICH item is
  audited is chosen uniformly at random among current deployment items, independently of
  that item's features or confidence. So each audited loss has conditional mean equal to
  the current deployment risk, and the betting rate uses only past audited losses. The
  wealth process therefore stays a test supermartingale under H0.
  (Auditing low-confidence items preferentially would bias audited losses upward and
  void the guarantee. Do not do that without importance weighting.)
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


# --------------------------------------------------------------------------- Tier 2
@dataclass
class BettingRiskMonitor:
    """Sequential test of H0: conditional risk <= eps_star at every time.

    Wealth M_t = prod_i (1 + lam_i * (z_i - eps_star)), z_i in [0, 1].
    lam_i is predictable (uses z_1..z_{i-1} only) and lies in [0, 1/(2*eps_star)],
    so every factor is positive. Rejects when M_t >= 1/delta.
    Betting rate: approximate growth-rate-optimal plug-in (Waudby-Smith & Ramdas 2024,
    as used by Timans et al. 2025), estimated on a sliding window of past losses.
    """
    eps_star: float
    delta: float = 0.05
    window: int | None = 200      # sliding window for the plug-in mean/variance
    burn_in: int = 20             # audited samples before betting starts (lam = 0)
    wealth: float = 1.0
    n: int = 0
    rejected_at: int | None = None
    _hist: list = field(default_factory=list)

    def _bet(self) -> float:
        if len(self._hist) < self.burn_in:
            return 0.0
        h = self._hist[-self.window:] if self.window else self._hist
        mu = float(np.mean(h))
        var = float(np.var(h))
        gap = mu - self.eps_star
        if gap <= 0:
            return 0.0
        lam = gap / (var + gap * gap + 1e-12)
        return float(min(max(lam, 0.0), 0.5 / self.eps_star))

    def update(self, z: float) -> bool:
        """Feed one audited loss in [0, 1]. Returns True once H0 is rejected."""
        lam = self._bet()                       # decided before seeing z
        self.wealth *= 1.0 + lam * (z - self.eps_star)
        self._hist.append(float(z))
        self.n += 1
        if self.rejected_at is None and self.wealth >= 1.0 / self.delta:
            self.rejected_at = self.n
        return self.rejected_at is not None


# --------------------------------------------------------------------------- Tier 1
@dataclass
class ATCEstimator:
    """Average Thresholded Confidence (Garg et al., ICLR 2022).

    Fit on labeled source validation data: pick threshold t so that the share of
    examples with confidence > t equals source accuracy. Estimated target accuracy =
    share of unlabeled target examples with confidence > t.
    """
    threshold: float = 0.5

    def fit(self, conf_src: np.ndarray, correct_src: np.ndarray) -> "ATCEstimator":
        acc = float(np.mean(correct_src))
        self.threshold = float(np.quantile(conf_src, 1.0 - acc))
        return self

    def estimate(self, conf_tgt: np.ndarray) -> float:
        return float(np.mean(conf_tgt > self.threshold))


# --------------------------------------------------------------------------- policy
@dataclass
class AuditPolicy:
    base_rate: float = 0.01       # always > 0: Tier 1 is blind to pure concept drift
    alert_rate: float = 0.10
    two_tier: bool = True
    check_every: int = 250        # Tier 1 re-estimates accuracy every this many items
    lookback: int = 500           # unlabeled items used per Tier 1 estimate
    reference: float | None = None  # relative trigger: alarm when the Tier 1 estimate falls
                                    # more than `tolerance` below its own value on the
                                    # provider's clean test data (cancels estimator bias).
                                    # None = absolute trigger at the violation level.


def run_stream(conf: np.ndarray, correct: np.ndarray, *, declared_acc: float,
               tolerance: float, delta: float, atc: ATCEstimator,
               policy: AuditPolicy, rng: np.random.Generator,
               label_delay: int = 0, trace: bool = False) -> dict:
    """Run the two-tier monitor over a deployment stream.

    conf[t]     model confidence (max softmax) for item t, always observed
    correct[t]  1 if the prediction is right; only read for audited items
    label_delay the label of an item audited at time t becomes available at t + label_delay
    Returns the rejection time (in stream items), labels used up to rejection, Tier 1
    alert times and, if trace=True, the audit-rate, ATC and wealth trajectories.

    Tier 1 reads only unlabeled confidences, so the audit schedule never depends on
    audited labels. We therefore draw the whole schedule first and then feed audited
    losses to Tier 2 in the order their labels arrive. This is identical in
    distribution to running both tiers online.
    """
    eps_star = (1.0 - declared_acc) + tolerance
    mon = BettingRiskMonitor(eps_star=eps_star, delta=delta)
    rate = policy.base_rate
    alerts, rates, ests, audit_times = [], [], [], []
    T = len(conf)
    B = policy.check_every
    for start in range(0, T, B):
        est = None
        if policy.two_tier and start >= policy.lookback:
            est = atc.estimate(conf[start - policy.lookback:start])
            level = (policy.reference - tolerance) if policy.reference is not None \
                else 1.0 - eps_star
            alarm = est < level
            if alarm and rate != policy.alert_rate:
                alerts.append(start)
            rate = policy.alert_rate if alarm else policy.base_rate
        rates.append((start, rate)); ests.append((start, est))
        idx = np.arange(start, min(start + B, T))
        audit_times.extend(idx[rng.random(len(idx)) < rate].tolist())

    wealth_trace = []
    rejected_at, used = None, 0
    for t in audit_times:                      # labels arrive in audit order + fixed delay
        arrival = t + label_delay
        if arrival >= T:
            break
        used += 1
        done = mon.update(1.0 - float(correct[t]))
        if trace:
            wealth_trace.append((arrival, mon.wealth))
        if done:
            rejected_at = int(arrival)
            break
    out = {"rejected_at": rejected_at, "audits": used, "alerts": alerts,
           "wealth": mon.wealth, "eps_star": eps_star}
    if trace:
        out.update({"rates": rates, "atc": ests, "wealth_trace": wealth_trace})
    return out
