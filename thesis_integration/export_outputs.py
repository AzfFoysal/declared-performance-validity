"""Export aligned 27-class predictions for the declared-validity case study.

Fixes three problems in the current evaluation path:
  1. PlantDoc labels were PlantDoc folder indices, never mapped to PlantVillage indices.
  2. The 11 PlantVillage-only classes were never masked before argmax.
  3. The PlantVillage lab accuracy was reported over 38 classes, so it could not be
     compared like-for-like with the 27-class field accuracy.

Output: an .npz with, for each split (val, lab = PlantVillage test, field = PlantDoc),
  <split>_conf     max softmax probability after masking to the 27 aligned classes
  <split>_correct  1 if the masked argmax equals the true (PlantVillage-index) label
  <split>_pred, <split>_true  PlantVillage class indices
Only images whose true class is one of the 27 aligned classes are kept, in every split.

Usage (on the machine with the data and checkpoint):
  python scripts/export_outputs.py --model hybrid \
      --weights checkpoints/hybrid_full_seed42.pth --data_dir data --out outputs.npz
Then:
  python case_study.py outputs.npz
"""
from __future__ import annotations

import argparse
import json
import os
import re

import numpy as np

# Appendix A of the thesis / AMLDS paper: PlantDoc folder -> PlantVillage class.
# PlantVillage names are matched after normalisation (see norm), so small differences
# such as "Corn_(maize)___Common_rust_" vs "Corn___Common_rust" are tolerated.
PLANTDOC_TO_PV = {
    "Apple Scab Leaf": "Apple___Apple_scab",
    "Apple leaf": "Apple___healthy",
    "Apple rust leaf": "Apple___Cedar_apple_rust",
    "Bell_pepper leaf spot": "Pepper,_bell___Bacterial_spot",
    "Bell_pepper leaf": "Pepper,_bell___healthy",
    "Blueberry leaf": "Blueberry___healthy",
    "Cherry leaf": "Cherry___healthy",
    "Corn Gray leaf spot": "Corn___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn rust leaf": "Corn___Common_rust",
    "Corn leaf blight": "Corn___Northern_Leaf_Blight",
    "Grape leaf black rot": "Grape___Black_rot",
    "Grape leaf": "Grape___healthy",
    "Peach leaf": "Peach___healthy",
    "Potato leaf early blight": "Potato___Early_blight",
    "Potato leaf late blight": "Potato___Late_blight",
    "Potato leaf": "Potato___healthy",
    "Raspberry leaf": "Raspberry___healthy",
    "Soyabean leaf": "Soybean___healthy",
    "Squash Powdery mildew leaf": "Squash___Powdery_mildew",
    "Strawberry leaf": "Strawberry___healthy",
    "Tomato leaf bacterial spot": "Tomato___Bacterial_spot",
    "Tomato Early blight leaf": "Tomato___Early_blight",
    "Tomato leaf late blight": "Tomato___Late_blight",
    "Tomato mold leaf": "Tomato___Leaf_Mold",
    "Tomato Septoria leaf spot": "Tomato___Septoria_leaf_spot",
    "Tomato leaf mosaic virus": "Tomato___Tomato_mosaic_virus",
    "Tomato leaf yellow virus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
}


# --------------------------------------------------------------- pure logic (tested)
def norm(name: str) -> str:
    """Lowercase, drop parenthesised text, keep letters and digits only."""
    name = re.sub(r"\(.*?\)", "", name.lower())
    return re.sub(r"[^a-z0-9]", "", name)


def resolve_mapping(pd_classes: list[str], pv_classes: list[str]) -> dict[int, int]:
    """Map PlantDoc folder index -> PlantVillage class index. Fails loudly on any gap."""
    pv_by_norm = {norm(c): i for i, c in enumerate(pv_classes)}
    pd_by_norm = {norm(c): i for i, c in enumerate(pd_classes)}
    out, missing = {}, []
    for pd_name, pv_name in PLANTDOC_TO_PV.items():
        pd_i, pv_i = pd_by_norm.get(norm(pd_name)), pv_by_norm.get(norm(pv_name))
        if pd_i is None or pv_i is None:
            missing.append((pd_name, pv_name, pd_i is not None, pv_i is not None))
        else:
            out[pd_i] = pv_i
    if missing:
        lines = "\n".join(f"  {a!r} -> {b!r} (PlantDoc found={c}, PlantVillage found={d})"
                          for a, b, c, d in missing)
        raise ValueError("Unmatched classes; fix PLANTDOC_TO_PV:\n" + lines)
    if len(set(out.values())) != len(out):
        raise ValueError("Two PlantDoc classes map to the same PlantVillage class.")
    return out


def aligned_mask(num_classes: int, aligned_pv: list[int]) -> np.ndarray:
    m = np.zeros(num_classes, dtype=bool)
    m[aligned_pv] = True
    return m


def masked_predictions(logits: np.ndarray, mask: np.ndarray):
    """Softmax over the aligned classes only; return (pred_index, confidence)."""
    z = np.where(mask[None, :], logits, -np.inf)
    z = z - z.max(axis=1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(axis=1, keepdims=True)
    return p.argmax(axis=1), p.max(axis=1)


def summarise(pred, true, conf):
    correct = (pred == true).astype(np.int8)
    return {"conf": conf.astype(np.float32), "correct": correct,
            "pred": pred.astype(np.int16), "true": true.astype(np.int16)}


# --------------------------------------------------------------- torch layer (run on GPU box)
def main():
    import torch
    from torch.utils.data import DataLoader
    from torchvision import datasets

    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from scripts.evaluate import load_model, PathLabelDataset
    from utils.augmentation import val_transform_cnn, val_transform_vit
    from utils.dataset_prep import HybridDataset

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["resnet", "vit", "hybrid"], required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--split_json", default=None)
    ap.add_argument("--out", default="outputs.npz")
    ap.add_argument("--batch_size", type=int, default=32)
    a = ap.parse_args()

    raw_dir = os.path.join(a.data_dir, "PlantVillage", "raw")
    pd_dir = os.path.join(a.data_dir, "PlantDoc", "plantdoc")
    split_json = a.split_json or os.path.join(a.data_dir, "splits", "plantvillage_split_seed42.json")

    pv = datasets.ImageFolder(raw_dir)
    pdoc = datasets.ImageFolder(pd_dir)
    pd_to_pv = resolve_mapping(pdoc.classes, pv.classes)
    aligned = sorted(pd_to_pv.values())
    mask = aligned_mask(len(pv.classes), aligned)
    print(f"Aligned classes: {len(aligned)} of {len(pv.classes)}")

    split = json.load(open(split_json))
    # create_splits.py writes "val"/"test"; older code expected "val_idx"/"test_idx"
    val_idx = split.get("val_idx", split.get("val"))
    test_idx = split.get("test_idx", split.get("test"))

    aligned_set = set(aligned)
    val_s = [pv.samples[i] for i in val_idx if pv.samples[i][1] in aligned_set]
    lab_s = [pv.samples[i] for i in test_idx if pv.samples[i][1] in aligned_set]
    fld_s = [(p, pd_to_pv[y]) for p, y in pdoc.samples if y in pd_to_pv]
    print(f"Images kept - val: {len(val_s)}, lab test: {len(lab_s)}, field: {len(fld_s)}")

    model, mtype = load_model(a.model, num_classes=len(pv.classes), weights_path=a.weights)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(dev).eval()

    class Thin:  # what HybridDataset expects
        def __init__(self, samples, loader):
            self.samples, self.loader = samples, loader

    def run(samples):
        if mtype == "hybrid":
            ds = HybridDataset(Thin(samples, pv.loader), val_transform_cnn, val_transform_vit)
        else:
            ds = PathLabelDataset(samples, pv.loader,
                                  val_transform_vit if mtype == "vit" else val_transform_cnn)
        logits, ys = [], []
        with torch.no_grad():
            for xb, yb in DataLoader(ds, batch_size=a.batch_size, shuffle=False, num_workers=4):
                if mtype == "hybrid":
                    out = model(xb[0].to(dev), xb[1].to(dev))
                else:
                    out = model(xb.to(dev))
                logits.append(out.float().cpu().numpy()); ys.append(np.asarray(yb))
        logits, ys = np.concatenate(logits), np.concatenate(ys)
        pred, conf = masked_predictions(logits, mask)
        return summarise(pred, ys, conf)

    res = {"val": run(val_s), "lab": run(lab_s), "field": run(fld_s)}
    flat = {f"{k}_{f}": v for k, d in res.items() for f, v in d.items()}
    np.savez(a.out, **flat, aligned_pv=np.array(aligned))
    for k in res:
        print(f"{k:>5}: accuracy on 27 aligned classes = {res[k]['correct'].mean():.4f} "
              f"(n = {len(res[k]['correct'])}), mean confidence = {res[k]['conf'].mean():.3f}")
    print(f"Saved {a.out}")


if __name__ == "__main__":
    main()
