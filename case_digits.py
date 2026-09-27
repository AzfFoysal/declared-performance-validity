"""Real-classifier case study: handwritten digits under real input corruptions.

A multilayer perceptron is trained on clean UCI handwritten digits (scikit-learn's
bundled copy, 1,797 8x8 images). Its declared accuracy is its accuracy on a held-out
clean test set. Deployment streams of 20,000 items draw test images with replacement:
5,000 clean items, then 15,000 items under one deployment condition. Corruptions are
applied afresh to every drawn item, so the classifier's inputs, confidences and errors
are real; only the order of arrival is simulated.

Conditions
  clean          no change (false-alarm check)
  noise_0.3      sensor noise, Gaussian sigma 0.3 on [0,1] pixel intensities
  blur_0.8       focus drift, Gaussian blur sigma 0.8 pixels
  blur_0.6       mild focus drift, sigma 0.6
  shift_1px      camera misalignment, image shifted one pixel right
  gradual_noise  noise sigma rising linearly 0 -> 0.4 over 10,000 items
  label_1_7      labeling-convention change: ground truth for digits 1 and 7 swapped;
                 inputs unchanged, so confidence cannot reveal it (concept drift)
Outputs: results/digits.csv, results/digits_trajectory.json, results/digits_meta.json
"""
import csv
import json
import os

import numpy as np
from scipy.ndimage import gaussian_filter, shift as nd_shift
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from monitor import ATCEstimator, AuditPolicy, run_stream

T, T0, T_RAMP = 20_000, 5_000, 10_000
TOL, DELTA, REPS = 0.02, 0.05, 300
POLICIES = {
    "fixed 1%": AuditPolicy(base_rate=0.01, two_tier=False),
    "two-tier 1% -> 10%": AuditPolicy(base_rate=0.01, alert_rate=0.10, two_tier=True),
    "fixed 10%": AuditPolicy(base_rate=0.10, two_tier=False),
}

X, y = load_digits(return_X_y=True)
X = X / 16.0
Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=0.5, stratify=y, random_state=0)
Xva, Xte, yva, yte = train_test_split(Xtmp, ytmp, test_size=0.6, stratify=ytmp, random_state=0)
clf = MLPClassifier((64,), max_iter=2000, random_state=0).fit(Xtr, ytr)
IMG = Xte.reshape(-1, 8, 8)
DECLARED = float(clf.score(Xte, yte))
atc = ATCEstimator().fit(clf.predict_proba(Xva).max(1), clf.predict(Xva) == yva)
ATC_REF = atc.estimate(clf.predict_proba(Xte).max(1))   # Tier 1 reading on clean test data
POLICIES["two-tier, relative trigger"] = AuditPolicy(base_rate=0.01, alert_rate=0.10,
                                                     two_tier=True, reference=ATC_REF)


def corrupt(imgs, kind, sev, rng):
    if kind == "noise":
        return np.clip(imgs + rng.normal(0, sev, imgs.shape), 0, 1) if sev > 0 else imgs
    if kind == "blur":
        return np.array([gaussian_filter(i, sev) for i in imgs])
    if kind == "shift":
        return np.array([nd_shift(i, (0, sev), order=0) for i in imgs])
    return imgs


def draw(cond, rng, n=T, t0=T0):
    idx = rng.integers(0, len(IMG), n)
    imgs, labels = IMG[idx].copy(), yte[idx].copy()
    post = np.arange(n) >= t0
    if cond in ("noise_0.3", "blur_0.8", "blur_0.6", "shift_1px"):
        kind, sev = cond.split("_")
        sev = float(sev.replace("px", ""))
        imgs[post] = corrupt(imgs[post], kind, sev, rng)
    elif cond == "gradual_noise":
        sev = np.clip((np.arange(n) - t0) / T_RAMP, 0, 1) * 0.4
        noise = rng.normal(0, 1, imgs.shape) * sev[:, None, None]
        imgs = np.clip(imgs + noise, 0, 1)
    elif cond == "label_1_7":
        lab = labels[post]
        labels[post] = np.where(lab == 1, 7, np.where(lab == 7, 1, lab))
    p = clf.predict_proba(imgs.reshape(n, -1))
    return p.max(1), p.argmax(1) == labels


def deployed_accuracy(cond, rng):
    """Accuracy after the change, from 50,000 fresh draws (end of ramp for gradual)."""
    conf, correct = draw(cond, rng, n=50_000, t0=0)
    if cond == "gradual_noise":  # severity at the end of the ramp
        idx = rng.integers(0, len(IMG), 50_000)
        imgs = np.clip(IMG[idx] + rng.normal(0, 0.4, IMG[idx].shape), 0, 1)
        p = clf.predict_proba(imgs.reshape(50_000, -1))
        conf, correct = p.max(1), p.argmax(1) == yte[idx]
    return float(correct.mean()), float(atc.estimate(conf))


CONDS = ["clean", "noise_0.3", "blur_0.8", "blur_0.6", "shift_1px", "gradual_noise", "label_1_7"]


def main():
    os.makedirs("results", exist_ok=True)
    rng = np.random.default_rng(20260927)
    viol = DECLARED - TOL
    meta = {"n_train": len(Xtr), "n_val": len(Xva), "n_test": len(Xte),
            "declared_accuracy": DECLARED, "violation_level": viol,
            "atc_threshold": atc.threshold, "atc_reference": ATC_REF,
            "val_accuracy": float(clf.score(Xva, yva))}
    rows = []
    for cond in CONDS:
        post_acc, atc_est = deployed_accuracy(cond, rng)
        # time the true accuracy first falls below the violation level
        if cond == "clean" or post_acc >= viol:
            t_cross = None
        elif cond == "gradual_noise":
            # find ramp severity where accuracy crosses the violation level
            sevs = np.linspace(0, 0.4, 41)
            accs = []
            for s in sevs:
                idx = rng.integers(0, len(IMG), 20_000)
                im = np.clip(IMG[idx] + rng.normal(0, s, IMG[idx].shape), 0, 1)
                accs.append((clf.predict(im.reshape(20_000, -1)) == yte[idx]).mean())
            s_cross = sevs[np.argmax(np.array(accs) < viol)]
            t_cross = int(T0 + s_cross / 0.4 * T_RAMP)
        else:
            t_cross = T0
        stats = {p: {"rej": 0, "early": 0, "delays": [], "labels": []} for p in POLICIES}
        for _ in range(REPS):
            conf, correct = draw(cond, rng)
            for pname, pol in POLICIES.items():
                out = run_stream(conf, correct, declared_acc=DECLARED, tolerance=TOL,
                                 delta=DELTA, atc=atc, policy=pol, rng=rng)
                s = stats[pname]; s["labels"].append(out["audits"])
                ta = out["rejected_at"]
                if ta is not None:
                    s["rej"] += 1
                    if t_cross is None or ta < t_cross:
                        s["early"] += 1
                    else:
                        s["delays"].append(ta - t_cross)
        for pname, s in stats.items():
            d = np.array(s["delays"]) if s["delays"] else np.array([np.nan])
            rows.append({"condition": cond, "policy": pname,
                         "deployed_accuracy": round(post_acc, 4), "atc_estimate": round(atc_est, 4),
                         "violated": t_cross is not None,
                         "false_alarm_rate": s["early"] / REPS,
                         "detection_rate": (s["rej"] - s["early"]) / REPS if t_cross is not None else None,
                         "median_delay": float(np.nanmedian(d)),
                         "q25_delay": float(np.nanpercentile(d, 25)) if s["delays"] else None,
                         "q75_delay": float(np.nanpercentile(d, 75)) if s["delays"] else None,
                         "mean_labels": float(np.mean(s["labels"]))})
            print(rows[-1])
    with open("results/digits.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    with open("results/digits_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # traced run: sensor noise, both label-saving policies
    tr_rng = np.random.default_rng(5)
    conf, correct = draw("noise_0.3", tr_rng)
    win = 500
    roll = np.convolve(correct.astype(float), np.ones(win) / win, mode="valid")
    traj = {"declared": DECLARED, "violation": viol, "shift_at": T0, "delta": DELTA,
            "rolling_acc_t": list(range(win - 1, T, 50)), "rolling_acc": roll[::50].tolist()}
    for pname in ("fixed 1%", "two-tier 1% -> 10%", "two-tier, relative trigger"):
        out = run_stream(conf, correct, declared_acc=DECLARED, tolerance=TOL, delta=DELTA,
                         atc=atc, policy=POLICIES[pname], rng=np.random.default_rng(3), trace=True)
        traj[pname] = {k: out[k] for k in ("rejected_at", "audits", "rates", "atc", "wealth_trace")}
    with open("results/digits_trajectory.json", "w") as f:
        json.dump(traj, f)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
