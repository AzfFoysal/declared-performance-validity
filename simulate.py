"""Simulation study for the two-tier declared-validity monitor.

Declared accuracy 96.3% (the thesis lab figure). Tolerance 2 points, so the declaration
counts as violated once deployed accuracy drops below 94.3% (eps_star = 0.057).
False-alarm level delta = 0.05. Each run is a stream of 20,000 deployment items.

Generative model (a stylised classifier):
  latent difficulty s ~ Normal(m_t, 1)
  confidence        c = sigmoid(A * s)                      (what the model reports)
  correct           ~ Bernoulli(sigmoid(A * s - b_t))       (b_t > 0 = concept drift)
Covariate shift moves m_t, which lowers both accuracy and confidence (Tier 1 can see it).
Concept drift raises b_t, which lowers accuracy but leaves confidence unchanged
(Tier 1 is blind to it, as Solozobov 2026 also finds).
"""
import csv
import numpy as np
from scipy.optimize import brentq
from monitor import ATCEstimator, AuditPolicy, run_stream

A = 2.0
DECLARED = 0.963
TOL = 0.02
DELTA = 0.05
EPS_STAR = (1 - DECLARED) + TOL
T, T0, T1 = 20_000, 5_000, 15_000
REPS = 300
MC = np.random.default_rng(0).standard_normal(400_000)


def sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def acc(m, b=0.0):
    return float(np.mean(sig(A * (m + MC) - b)))


M0 = brentq(lambda m: acc(m) - DECLARED, -5, 10)          # source: 96.3%
M_FIELD = brentq(lambda m: acc(m) - 0.78, -5, 10)         # field: 78%
M_MILD = brentq(lambda m: acc(m) - 0.925, -5, 10)         # mild: 92.5%
M_EDGE = brentq(lambda m: acc(m) - (1 - EPS_STAR), -5, 10)  # exactly on H0 boundary
B_CONCEPT = brentq(lambda b: acc(M0, b) - 0.78, 0, 20)    # concept drift to 78%


def schedule(kind):
    t = np.arange(T)
    m = np.full(T, M0)
    b = np.zeros(T)
    if kind == "no_shift":
        pass
    elif kind == "boundary":
        m[:] = M_EDGE
    elif kind == "abrupt_covariate":
        m[T0:] = M_FIELD
    elif kind == "gradual_covariate":
        frac = np.clip((t - T0) / (T1 - T0), 0, 1)
        m = M0 + frac * (M_FIELD - M0)
    elif kind == "mild_covariate":
        m[T0:] = M_MILD
    elif kind == "abrupt_concept":
        b[T0:] = B_CONCEPT
    return m, b


def true_risk(m, b):
    # risk as a function of (m, b); evaluated on a coarse grid for speed
    out = np.empty(len(m))
    cache = {}
    for i, (mi, bi) in enumerate(zip(m, b)):
        k = (round(mi, 3), round(bi, 3))
        if k not in cache:
            cache[k] = 1 - acc(mi, bi)
        out[i] = cache[k]
    return out


def draw_stream(m, b, rng):
    s = m + rng.standard_normal(len(m))
    conf = sig(A * s)
    correct = rng.random(len(m)) < sig(A * s - b)
    return conf, correct


POLICIES = {
    "fixed 1%": AuditPolicy(base_rate=0.01, two_tier=False),
    "two-tier 1% -> 10%": AuditPolicy(base_rate=0.01, alert_rate=0.10, two_tier=True),
    "fixed 10%": AuditPolicy(base_rate=0.10, two_tier=False),
}
SCENARIOS = ["no_shift", "boundary", "abrupt_covariate", "gradual_covariate",
             "mild_covariate", "abrupt_concept"]


def main():
    rng = np.random.default_rng(2026)
    # Tier 1 is fitted once on a labeled source validation set (as in practice)
    s_val = M0 + rng.standard_normal(5_000)
    atc = ATCEstimator().fit(sig(A * s_val), rng.random(5_000) < sig(A * s_val))

    rows = []
    for sc in SCENARIOS:
        m, b = schedule(sc)
        r = true_risk(m, b)
        above = np.nonzero(r > EPS_STAR)[0]
        t_cross = int(above[0]) if len(above) else None
        if sc in ("no_shift", "boundary"):   # H0 holds throughout: any alarm is false
            t_cross = None
        for pname, pol in POLICIES.items():
            rej, early, delays, audits = 0, 0, [], []
            for _ in range(REPS):
                conf, correct = draw_stream(m, b, rng)
                out = run_stream(conf, correct, declared_acc=DECLARED, tolerance=TOL,
                                 delta=DELTA, atc=atc, policy=pol, rng=rng)
                audits.append(out["audits"])
                ta = out["rejected_at"]
                if ta is not None:
                    rej += 1
                    if t_cross is None or ta < t_cross:
                        early += 1
                    else:
                        delays.append(ta - t_cross)
            rows.append({
                "scenario": sc,
                "policy": pname,
                "false_alarm_rate": early / REPS,
                "detection_rate": (rej - early) / REPS if t_cross is not None else None,
                "median_delay_items": float(np.median(delays)) if delays else None,
                "mean_audits": float(np.mean(audits)),
            })
            print(rows[-1])
    import os; os.makedirs("results", exist_ok=True)
    with open("results/simulation_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    print("source acc check", acc(M0), "field", acc(M_FIELD), "edge", acc(M_EDGE),
          "concept", acc(M0, B_CONCEPT), "eps*", EPS_STAR, "ATC thr", atc.threshold)


if __name__ == "__main__":
    main()
