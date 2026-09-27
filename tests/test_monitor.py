"""Checks for the two-tier monitor and the 27-class export logic.

Run with:  python -m pytest -q
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "thesis_integration"))

from monitor import ATCEstimator, AuditPolicy, BettingRiskMonitor, run_stream  # noqa: E402
from export_outputs import aligned_mask, masked_predictions, resolve_mapping  # noqa: E402


def test_false_alarm_rate_under_null_is_below_delta():
    # Losses exactly at the violation level: H0 holds on its boundary.
    rng = np.random.default_rng(0)
    eps, delta, reps = 0.057, 0.05, 400
    alarms = 0
    for _ in range(reps):
        mon = BettingRiskMonitor(eps_star=eps, delta=delta)
        for z in rng.random(2000) < eps:
            if mon.update(float(z)):
                alarms += 1
                break
    assert alarms / reps <= delta


def test_detects_large_violation():
    rng = np.random.default_rng(1)
    mon = BettingRiskMonitor(eps_star=0.057, delta=0.05)
    rejected = any(mon.update(float(z)) for z in rng.random(1000) < 0.22)
    assert rejected


def test_bets_are_nonnegative_and_bounded():
    mon = BettingRiskMonitor(eps_star=0.05, delta=0.05, burn_in=1)
    for z in [1.0] * 50:
        lam = mon._bet()
        assert 0.0 <= lam <= 0.5 / 0.05
        mon.update(z)


def test_atc_recovers_source_accuracy():
    rng = np.random.default_rng(2)
    conf = rng.random(5000)
    correct = rng.random(5000) < conf
    atc = ATCEstimator().fit(conf, correct)
    assert abs(atc.estimate(conf) - correct.mean()) < 0.01


def test_two_tier_uses_fewer_labels_than_fixed_high_rate():
    rng = np.random.default_rng(3)
    sig = lambda x: 1 / (1 + np.exp(-x))  # noqa: E731
    s = np.concatenate([1.9 + rng.standard_normal(3000), 0.7 + rng.standard_normal(6000)])
    conf, correct = sig(2 * s), rng.random(len(s)) < sig(2 * s)
    sv = 1.9 + rng.standard_normal(3000)
    atc = ATCEstimator().fit(sig(2 * sv), rng.random(3000) < sig(2 * sv))
    kw = dict(declared_acc=float(correct[:3000].mean()), tolerance=0.02, delta=0.05, atc=atc)
    two = run_stream(conf, correct, policy=AuditPolicy(), rng=np.random.default_rng(4), **kw)
    hi = run_stream(conf, correct, policy=AuditPolicy(base_rate=0.10, two_tier=False),
                    rng=np.random.default_rng(4), **kw)
    assert two["rejected_at"] is not None and hi["rejected_at"] is not None
    assert two["audits"] < hi["audits"]


PV = ("Apple___Apple_scab|Apple___Black_rot|Apple___Cedar_apple_rust|Apple___healthy|"
      "Blueberry___healthy|Cherry_(including_sour)___Powdery_mildew|Cherry_(including_sour)___healthy|"
      "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot|Corn_(maize)___Common_rust_|"
      "Corn_(maize)___Northern_Leaf_Blight|Corn_(maize)___healthy|Grape___Black_rot|"
      "Grape___Esca_(Black_Measles)|Grape___Leaf_blight_(Isariopsis_Leaf_Spot)|Grape___healthy|"
      "Orange___Haunglongbing_(Citrus_greening)|Peach___Bacterial_spot|Peach___healthy|"
      "Pepper,_bell___Bacterial_spot|Pepper,_bell___healthy|Potato___Early_blight|Potato___Late_blight|"
      "Potato___healthy|Raspberry___healthy|Soybean___healthy|Squash___Powdery_mildew|"
      "Strawberry___Leaf_scorch|Strawberry___healthy|Tomato___Bacterial_spot|Tomato___Early_blight|"
      "Tomato___Late_blight|Tomato___Leaf_Mold|Tomato___Septoria_leaf_spot|"
      "Tomato___Spider_mites Two-spotted_spider_mite|Tomato___Target_Spot|"
      "Tomato___Tomato_Yellow_Leaf_Curl_Virus|Tomato___Tomato_mosaic_virus|Tomato___healthy").split("|")
PD = ["Apple Scab Leaf", "Apple leaf", "Apple rust leaf", "Bell_pepper leaf spot", "Bell_pepper leaf",
      "Blueberry leaf", "Cherry leaf", "Corn Gray leaf spot", "Corn rust leaf", "Corn leaf blight",
      "grape leaf black rot", "grape leaf", "Peach leaf", "Potato leaf early blight",
      "Potato leaf late blight", "Potato leaf", "Raspberry leaf", "Soyabean leaf",
      "Squash Powdery mildew leaf", "Strawberry leaf", "Tomato leaf bacterial spot",
      "Tomato Early blight leaf", "Tomato leaf late blight", "Tomato mold leaf",
      "Tomato Septoria leaf spot", "Tomato leaf mosaic virus", "Tomato leaf yellow virus", "Tomato leaf"]


def test_mapping_covers_27_classes_one_to_one():
    m = resolve_mapping(sorted(PD), sorted(PV))
    assert len(m) == 27 and len(set(m.values())) == 27


def test_masked_predictions_never_pick_unaligned_classes():
    m = resolve_mapping(sorted(PD), sorted(PV))
    mask = aligned_mask(38, sorted(m.values()))
    logits = np.random.default_rng(5).normal(size=(200, 38)) * 5
    pred, conf = masked_predictions(logits, mask)
    assert mask[pred].all()
    assert np.all((conf > 0) & (conf <= 1))
