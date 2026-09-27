"""All simulation experiments reported in the paper.

Outputs (results/):
  main.csv          6 scenarios x 3 audit policies, 300 runs each
  sensitivity.csv   audit floor x tolerance, abrupt covariate shift (+ boundary false alarms)
  label_delay.csv   label delay 0..5000 items, abrupt covariate shift
  effect_size.csv   post-shift accuracy 0.92..0.70
  delta.csv         false-alarm level 0.01 / 0.05 / 0.10
  trajectory.json   one traced run per policy, abrupt covariate shift
"""
import csv
import json
import os
from multiprocessing import Pool

import numpy as np
from scipy.optimize import brentq

from monitor import ATCEstimator, AuditPolicy, run_stream
from simulate import (A, DECLARED, TOL, DELTA, T, T0, acc, sig, schedule, true_risk,
                      draw_stream, POLICIES, SCENARIOS, M0)

OUT = "results"
os.makedirs(OUT, exist_ok=True)
REPS = 300


def fit_atc(seed=2026):
    rng = np.random.default_rng(seed)
    s = M0 + rng.standard_normal(5000)
    return ATCEstimator().fit(sig(A * s), rng.random(5000) < sig(A * s))


ATC = fit_atc()


def m_for(accuracy):
    return brentq(lambda m: acc(m) - accuracy, -5, 10)


def one_cell(args):
    """Run `reps` streams for one configuration and summarise."""
    (m, b, t_cross, pol, tol, delta, lag, reps, seed) = args
    rng = np.random.default_rng(seed)
    rej = early = 0
    delays, labels = [], []
    for _ in range(reps):
        conf, correct = draw_stream(m, b, rng)
        out = run_stream(conf, correct, declared_acc=DECLARED, tolerance=tol, delta=delta,
                         atc=ATC, policy=pol, rng=rng, label_delay=lag)
        labels.append(out["audits"])
        ta = out["rejected_at"]
        if ta is not None:
            rej += 1
            if t_cross is None or ta < t_cross:
                early += 1
            else:
                delays.append(ta - t_cross)
    d = np.array(delays) if delays else np.array([np.nan])
    return {
        "false_alarm_rate": early / reps,
        "detection_rate": None if t_cross is None else (rej - early) / reps,
        "median_delay": float(np.nanmedian(d)),
        "q25_delay": float(np.nanpercentile(d, 25)),
        "q75_delay": float(np.nanpercentile(d, 75)),
        "mean_labels": float(np.mean(labels)),
    }


def shift_schedule(post_acc=None, kind="abrupt_covariate"):
    if post_acc is None:
        return schedule(kind)
    m = np.full(T, M0); m[T0:] = m_for(post_acc)
    return m, np.zeros(T)


def t_cross_of(kind, m, b, tol):
    if kind in ("no_shift", "boundary"):
        return None
    r = true_risk(m, b)
    above = np.nonzero(r > (1 - DECLARED) + tol)[0]
    return int(above[0]) if len(above) else None


def write(name, rows):
    with open(os.path.join(OUT, name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    print("wrote", name, len(rows), "rows")


def main():
    pool = Pool(os.cpu_count())
    seed = iter(range(10_000, 99_999))

    # ---- main table
    jobs, keys = [], []
    for sc in SCENARIOS:
        m, b = schedule(sc)
        tc = t_cross_of(sc, m, b, TOL)
        for pname, pol in POLICIES.items():
            jobs.append((m, b, tc, pol, TOL, DELTA, 0, REPS, next(seed)))
            keys.append({"scenario": sc, "policy": pname})
    write("main.csv", [{**k, **r} for k, r in zip(keys, pool.map(one_cell, jobs))])

    # ---- sensitivity: audit floor x tolerance (two-tier, alert rate 10%)
    jobs, keys = [], []
    m, b = schedule("abrupt_covariate")
    for tol in (0.01, 0.02, 0.03):
        tc = t_cross_of("abrupt_covariate", m, b, tol)
        edge = np.full(T, m_for(DECLARED - tol))
        for floor in (0.005, 0.01, 0.02, 0.05):
            pol = AuditPolicy(base_rate=floor, alert_rate=0.10, two_tier=True)
            jobs.append((m, b, tc, pol, tol, DELTA, 0, 200, next(seed)))
            keys.append({"tolerance": tol, "floor": floor, "scenario": "abrupt_covariate"})
            jobs.append((edge, np.zeros(T), None, pol, tol, DELTA, 0, 200, next(seed)))
            keys.append({"tolerance": tol, "floor": floor, "scenario": "boundary"})
    write("sensitivity.csv", [{**k, **r} for k, r in zip(keys, pool.map(one_cell, jobs))])

    # ---- label delay
    jobs, keys = [], []
    tc = t_cross_of("abrupt_covariate", m, b, TOL)
    for lag in (0, 500, 1000, 2000, 5000):
        for pname in ("two-tier 1% -> 10%", "fixed 10%"):
            jobs.append((m, b, tc, POLICIES[pname], TOL, DELTA, lag, 200, next(seed)))
            keys.append({"label_delay": lag, "policy": pname})
    write("label_delay.csv", [{**k, **r} for k, r in zip(keys, pool.map(one_cell, jobs))])

    # ---- effect size
    jobs, keys = [], []
    for post in (0.92, 0.90, 0.88, 0.85, 0.80, 0.75, 0.70):
        mm, bb = shift_schedule(post)
        tc = t_cross_of("x", mm, bb, TOL)
        for pname, pol in POLICIES.items():
            jobs.append((mm, bb, tc, pol, TOL, DELTA, 0, 200, next(seed)))
            keys.append({"post_accuracy": post, "policy": pname})
    write("effect_size.csv", [{**k, **r} for k, r in zip(keys, pool.map(one_cell, jobs))])

    # ---- false-alarm level
    jobs, keys = [], []
    edge = np.full(T, m_for(DECLARED - TOL))
    for delta in (0.01, 0.05, 0.10):
        pol = POLICIES["two-tier 1% -> 10%"]
        jobs.append((m, b, tc, pol, TOL, delta, 0, 200, next(seed)))
        keys.append({"delta": delta, "scenario": "abrupt_covariate"})
        jobs.append((edge, np.zeros(T), None, pol, TOL, delta, 0, 300, next(seed)))
        keys.append({"delta": delta, "scenario": "boundary"})
    write("delta.csv", [{**k, **r} for k, r in zip(keys, pool.map(one_cell, jobs))])

    # ---- one traced run per policy
    rng = np.random.default_rng(7)
    conf, correct = draw_stream(m, b, rng)
    true_acc = 1 - true_risk(m, b)
    traj = {"true_acc": true_acc[::50].tolist(), "t": list(range(0, T, 50)),
            "declared": DECLARED, "violation": DECLARED - TOL, "delta": DELTA, "shift_at": T0}
    for pname in ("fixed 1%", "two-tier 1% -> 10%"):
        out = run_stream(conf, correct, declared_acc=DECLARED, tolerance=TOL, delta=DELTA,
                         atc=ATC, policy=POLICIES[pname], rng=np.random.default_rng(11),
                         trace=True)
        traj[pname] = {k: out[k] for k in ("rejected_at", "audits", "rates", "atc", "wealth_trace")}
    with open(os.path.join(OUT, "trajectory.json"), "w") as f:
        json.dump(traj, f)
    print("wrote trajectory.json")


if __name__ == "__main__":
    main()
