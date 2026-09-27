"""Case study: run the two-tier monitor on your own model outputs.

STEP 1 - export predictions from your trained hybrid CNN-ViT (PyTorch), e.g.:

    import numpy as np, torch
    def dump(model, loader, keep_classes=None):
        conf, correct = [], []
        model.eval()
        with torch.no_grad():
            for x, y in loader:
                p = torch.softmax(model(x.cuda()), dim=1).cpu()
                c, yhat = p.max(dim=1)
                conf.append(c.numpy()); correct.append((yhat == y).numpy())
        return np.concatenate(conf), np.concatenate(correct)

    # Use ONLY the 27 classes aligned between PlantVillage and PlantDoc for all three
    # splits, so the lab and field accuracies are measured on the same label set.
    val_c, val_y   = dump(model, plantvillage_val_loader_27)    # fits Tier 1
    lab_c, lab_y   = dump(model, plantvillage_test_loader_27)   # the "declared" period
    fld_c, fld_y   = dump(model, plantdoc_test_loader_27)       # field deployment
    np.savez("outputs.npz", val_conf=val_c, val_correct=val_y,
             lab_conf=lab_c, lab_correct=lab_y, field_conf=fld_c, field_correct=fld_y)

STEP 2 - run:
    python case_study.py outputs.npz --tolerance 0.02 --delta 0.05

The script declares accuracy = lab test accuracy (27 classes), builds deployment streams
of lab images followed by field images (resampled, since PlantDoc is small), and reports
how quickly each audit policy invalidates the declaration and how many labels it needed.
"""
import argparse
import numpy as np
from monitor import ATCEstimator, AuditPolicy, run_stream

POLICIES = {
    "fixed 1%": AuditPolicy(base_rate=0.01, two_tier=False),
    "two-tier 1% -> 10%": AuditPolicy(base_rate=0.01, alert_rate=0.10, two_tier=True),
    "fixed 10%": AuditPolicy(base_rate=0.10, two_tier=False),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--tolerance", type=float, default=0.02)
    ap.add_argument("--delta", type=float, default=0.05)
    ap.add_argument("--lab_items", type=int, default=5000)
    ap.add_argument("--field_items", type=int, default=15000)
    ap.add_argument("--reps", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    d = np.load(a.npz)
    declared = float(d["lab_correct"].mean())
    field_acc = float(d["field_correct"].mean())
    atc = ATCEstimator().fit(d["val_conf"], d["val_correct"])
    print(f"Declared (lab, 27 classes): {declared:.3f}   Field: {field_acc:.3f}")
    print(f"Violation level: accuracy < {declared - a.tolerance:.3f} (delta = {a.delta})")
    print(f"ATC estimate on field images: {atc.estimate(d['field_conf']):.3f}\n")

    rng = np.random.default_rng(a.seed)
    nl, nf = len(d["lab_conf"]), len(d["field_conf"])
    print(f"{'policy':<22}{'detected':>10}{'false alarm':>13}{'median delay':>14}{'mean audits':>13}")
    for name, pol in POLICIES.items():
        det, fa, delays, audits = 0, 0, [], []
        for _ in range(a.reps):
            li = rng.integers(0, nl, a.lab_items)
            fi = rng.integers(0, nf, a.field_items)
            conf = np.concatenate([d["lab_conf"][li], d["field_conf"][fi]])
            corr = np.concatenate([d["lab_correct"][li], d["field_correct"][fi]])
            out = run_stream(conf, corr, declared_acc=declared, tolerance=a.tolerance,
                             delta=a.delta, atc=atc, policy=pol, rng=rng)
            audits.append(out["audits"])
            t = out["rejected_at"]
            if t is not None:
                if t < a.lab_items:
                    fa += 1
                else:
                    det += 1; delays.append(t - a.lab_items)
        med = f"{np.median(delays):.0f}" if delays else "-"
        print(f"{name:<22}{det / a.reps:>10.2f}{fa / a.reps:>13.2f}{med:>14}{np.mean(audits):>13.0f}")
    print("\nDelay = field images processed before the declaration was invalidated.")


if __name__ == "__main__":
    main()
