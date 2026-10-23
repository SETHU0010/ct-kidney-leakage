"""External validation of the Obj1 models on two public Stone / no-stone CT cohorts.

Standalone port of notebook Phase 18.5 (v12.8) so it runs on CPU outside the notebook.
Reads  : Obj1/tables/manifest_final.csv (hashes of the 12,446 main images), Obj1/checkpoints,
         Obj1/deploy/student_int8.onnx, Obj1/deploy/deploy_config.json
Writes : Obj1/external_validation/  (nothing in Obj1/tables or RESULTS.json is touched)

Scoring is the Stone-vs-Normal sub-task, exactly as in the notebook:
  normal_recall = P(pred == Normal | Normal)   <- the protocol-shortcut probe
  specificity   = P(pred != Stone  | Normal)
  stone_sens    = P(pred == Stone  | Stone)
CIs resample groups (patient ID from the filename where there is one, DAP hash groups otherwise).
External images that are byte-identical or DAP near-duplicates (dHash<=4 and pHash<=7) of any
main-dataset image are excluded and listed.

Usage:  python external_validation.py [--limit N]     (--limit N = N images per cohort, smoke test)
"""
import argparse, hashlib, json, re, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import timm
import onnxruntime as ort
from PIL import Image
from scipy.fft import dct
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import roc_auc_score
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "external_validation"
EXT_ROOT = Path.home() / "datasets"
CLASSES = ("Cyst", "Normal", "Stone", "Tumor")
C2I = {c: i for i, c in enumerate(CLASSES)}
STONE, NORMAL = C2I["Stone"], C2I["Normal"]
DHASH_T, PHASH_T = 4, 7          # FINAL_DTHR / FINAL_PTHR in RESULTS.json
IMG = 224
VALID_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
eval_tf = transforms.Compose([transforms.Resize((IMG, IMG)), transforms.ToTensor(),
                              transforms.Normalize(MEAN, STD)])

COHORTS = {
    "yildirim": {"desc": "Yildirim et al. 2021, coronal CT",
                 "root": EXT_ROOT / "Kidney_stone_detection" / "Dataset",
                 "folders": {"Train/Kidney_stone": "Stone", "Train/Normal": "Normal",
                             "Test/Kidney_stone": "Stone", "Test/Normal": "Normal"},
                 "patient_regex": None},
    "axial": {"desc": "'Kindey Stone Dataset', Original/ only; Non-Stone scored as Normal",
              "root": EXT_ROOT / "axial_stone" / "Kindey Stone Dataset" / "Original",
              "folders": {"Stone": "Stone", "Non-Stone": "Normal"},
              "patient_regex": r"^(P\d+)_"},
}


# ---------------------------------------------------------------- hashing (identical to notebook Phase 2.1)
def _gray(path, size):
    return np.asarray(Image.open(path).convert("L").resize((size, size), Image.LANCZOS), dtype=np.float32)

def dhash_bits(path, n=8):
    a = _gray(path, n + 1)[:n, :n + 1]
    return (a[:, 1:] > a[:, :-1]).flatten().astype(np.uint8)

def phash_bits(path, n=8, hf=4):
    a = _gray(path, n * hf)
    d = dct(dct(a, axis=0, norm="ortho"), axis=1, norm="ortho")[:n, :n]
    return (d > np.median(d[1:, 1:])).flatten().astype(np.uint8)

def md5_of(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for ch in iter(lambda: f.read(1 << 16), b""):
            h.update(ch)
    return h.hexdigest()

def hash_one(path):
    return dhash_bits(path), phash_bits(path), md5_of(path)

def hamming(a, b):
    """(n,64) x (m,64) uint8 bit arrays -> (n,m) Hamming distances."""
    a, b = a.astype(np.int32), b.astype(np.int32)
    return a.sum(1)[:, None] + b.sum(1)[None, :] - 2 * (a @ b.T)

def bits(s):
    return np.frombuffer(s.encode(), dtype=np.uint8) - 48


# ---------------------------------------------------------------- models (identical to notebook Phase 5.1)
def to_tokens(x):
    if x.dim() == 3:
        return x
    if x.shape[1] == x.shape[2] and x.shape[3] >= x.shape[1]:
        B, H, W, C = x.shape
        return x.reshape(B, H * W, C)
    return x.flatten(2).transpose(1, 2)

class TimmBaseline(nn.Module):
    def __init__(s, name, num_classes=4, dropout=0.3):
        super().__init__()
        s.backbone = timm.create_model(name, pretrained=False, num_classes=0, drop_rate=0.0)
        with torch.no_grad():
            d = s.backbone(torch.zeros(1, 3, IMG, IMG)).shape[-1]
        s.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d, num_classes))
    def forward(s, x):
        return s.head(s.backbone(x))

class RenalDualNet(nn.Module):
    def __init__(s, num_classes=4, fusion_dim=512, head_dropout=0.3):
        super().__init__()
        s.local_backbone = timm.create_model("convnext_tiny.fb_in1k", pretrained=False, num_classes=0)
        s.global_backbone = timm.create_model("swin_tiny_patch4_window7_224.ms_in1k", pretrained=False, num_classes=0)
        with torch.no_grad():
            d = torch.zeros(1, 3, IMG, IMG)
            dl = to_tokens(s.local_backbone.forward_features(d)).shape[-1]
            dg = to_tokens(s.global_backbone.forward_features(d)).shape[-1]
        s.mlp = nn.Sequential(nn.Linear(dl + dg, fusion_dim), nn.GELU(), nn.Dropout(0.1))
        s.head = nn.Sequential(nn.LayerNorm(fusion_dim), nn.Dropout(head_dropout), nn.Linear(fusion_dim, num_classes))
    def forward(s, x):
        pl = to_tokens(s.local_backbone.forward_features(x)).mean(1)
        pg = to_tokens(s.global_backbone.forward_features(x)).mean(1)
        return s.head(s.mlp(torch.cat([pl, pg], dim=1)))

def torch_logits(model, paths, bs=32):
    model.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(paths), bs):
            x = torch.stack([eval_tf(Image.open(p).convert("RGB")) for p in paths[s:s + bs]])
            out.append(model(x).numpy())
            print(f"    {min(s + bs, len(paths))}/{len(paths)}", end="\r", flush=True)
    print()
    return np.concatenate(out)

def onnx_logits(onnx_path, paths):
    so = ort.SessionOptions(); so.intra_op_num_threads = 4
    sess = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])
    nm = sess.get_inputs()[0].name
    return np.stack([sess.run(None, {nm: eval_tf(Image.open(p).convert("RGB")).unsqueeze(0).numpy()})[0][0]
                     for p in paths])

def softmax(lg, T=1.0):
    z = lg / T
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


# ---------------------------------------------------------------- metrics
def binary(y, prob, pred):
    st, nm = y == STONE, y == NORMAL
    sens = float((pred[st] == STONE).mean()) if st.any() else float("nan")
    spec = float((pred[nm] != STONE).mean()) if nm.any() else float("nan")
    auc = float(roc_auc_score(st[st | nm], prob[st | nm, STONE])) if st.any() and nm.any() else float("nan")
    return {"n": int((st | nm).sum()), "stone_sensitivity": sens, "specificity": spec,
            "normal_recall": float((pred[nm] == NORMAL).mean()) if nm.any() else float("nan"),
            "balanced_accuracy": (sens + spec) / 2, "auc_p_stone": auc}

def binary_ci(y, prob, pred, groups, n_boot=1000, seed=42,
              keys=("balanced_accuracy", "auc_p_stone", "normal_recall", "stone_sensitivity")):
    rng = np.random.RandomState(seed)
    ug = np.unique(groups)
    idx = {g: np.where(groups == g)[0] for g in ug}
    samp = {k: [] for k in keys}
    for _ in range(n_boot):
        ix = np.concatenate([idx[g] for g in rng.choice(ug, len(ug), replace=True)])
        if not ((y[ix] == STONE).any() and (y[ix] == NORMAL).any()):
            continue
        b = binary(y[ix], prob[ix], pred[ix])
        for k in keys:
            if np.isfinite(b[k]):
                samp[k].append(b[k])
    return {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if v else (np.nan, np.nan)
            for k, v in samp.items()}


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="images per cohort (smoke test)")
    ap.add_argument("--skip-teacher", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    torch.set_num_threads(max(1, torch.get_num_threads()))

    # hashes are 64-char bit strings: read as str, or pandas turns them into ints and drops leading zeros
    man = pd.read_csv(ROOT / "tables" / "manifest_final.csv", dtype={"dhash_hex": str, "phash_hex": str})
    assert man.dhash_hex.str.len().eq(64).all() and man.phash_hex.str.len().eq(64).all()
    DH = np.stack([bits(s) for s in man.dhash_hex]); PH = np.stack([bits(s) for s in man.phash_hex])
    md5_main = dict(zip(man.md5, man.split))
    print(f"main manifest: {len(man):,} images")

    dep = json.loads((ROOT / "deploy" / "deploy_config.json").read_text())
    models = {}
    sw = ROOT / "checkpoints" / "baseline_swin_tiny_patch4_window7_224.pt"
    m = TimmBaseline("swin_tiny_patch4_window7_224.ms_in1k")
    m.load_state_dict(torch.load(sw, map_location="cpu", weights_only=False)["state_dict"])
    models["Swin-T (seed 42)"] = {"fn": lambda p, m=m: torch_logits(m, p), "T": 1.0}
    if not args.skip_teacher:
        t = RenalDualNet()
        t.load_state_dict(torch.load(ROOT / "checkpoints" / "RenalDualNet_s42.pt", map_location="cpu",
                                     weights_only=False)["state_dict"])
        models["RenalDualNet (seed 42)"] = {"fn": lambda p, t=t: torch_logits(t, p), "T": 1.0}
    onnx_p = ROOT / "deploy" / dep["model_file"]
    models[f"Student ({onnx_p.stem}, T={dep['temperature']})"] = {
        "fn": lambda p: onnx_logits(onnx_p, p), "T": float(dep["temperature"])}

    rows, summary = [], {}
    for cname, cfg in COHORTS.items():
        print(f"\n=== {cname}: {cfg['desc']}")
        recs = []
        for sub, cls in cfg["folders"].items():
            d = cfg["root"] / sub
            for p in sorted(d.iterdir()) if d.exists() else []:
                if p.suffix.lower() in VALID_EXT:
                    recs.append({"path": str(p), "filename": p.name, "class_name": cls, "label": C2I[cls],
                                 "ext_split": sub.split("/")[0] if "/" in sub else "all"})
        edf = pd.DataFrame(recs)
        if edf.empty:
            summary[cname] = {"status": "not_found", "root": str(cfg["root"])}; print("  not found"); continue
        if args.limit:   # stratified smoke-test subset
            edf = pd.concat([g.sample(min(len(g), args.limit // 2), random_state=0)
                             for _, g in edf.groupby("class_name")]).reset_index(drop=True)
        print(f"  images {len(edf):,}  {edf.class_name.value_counts().to_dict()}")

        t0 = time.time()
        hs = [hash_one(p) for p in edf.path]
        DHe = np.stack([h[0] for h in hs]); PHe = np.stack([h[1] for h in hs]); edf["md5"] = [h[2] for h in hs]
        print(f"  hashed in {time.time() - t0:.0f}s")

        # --- overlap with the main dataset ---
        edf["main_md5_split"] = edf.md5.map(md5_main)
        near_idx = np.full(len(edf), -1, dtype=np.int64)
        near_d = np.full(len(edf), -1); near_p = np.full(len(edf), -1)
        for s in range(0, len(edf), 512):
            hd, hp = hamming(DHe[s:s + 512], DH), hamming(PHe[s:s + 512], PH)
            ok = (hd <= DHASH_T) & (hp <= PHASH_T)
            r = ok.any(1)
            first = np.where(ok, 0, 1).argmin(1)  # first matching column
            near_idx[s:s + 512][r] = first[r]
            near_d[s:s + 512][r] = hd[np.where(r)[0], first[r]]; near_p[s:s + 512][r] = hp[np.where(r)[0], first[r]]
        has_near = near_idx >= 0
        overlap = edf.main_md5_split.notna().values | has_near
        edf["main_neardup_file"] = np.where(has_near, man.filename.values[np.maximum(near_idx, 0)], None)
        edf["main_neardup_class"] = np.where(has_near, man.class_name.values[np.maximum(near_idx, 0)], None)
        edf["dhash_dist"], edf["phash_dist"] = near_d, near_p
        print(f"  overlap with main: byte-identical {int(edf.main_md5_split.notna().sum())}, "
              f"DAP near-duplicate {int(has_near.sum())} -> excluded {int(overlap.sum())}")
        edf[overlap].to_csv(OUT / f"{cname}_excluded.csv", index=False)
        # Also report the CLOSEST main image per external image, to show how far from the threshold we are.
        mind = np.concatenate([hamming(DHe[s:s + 512], DH).min(1) for s in range(0, len(edf), 512)])
        print(f"  min dHash distance to main: median {np.median(mind):.0f}, 5th pct {np.percentile(mind, 5):.0f}")
        edf = edf[~overlap].reset_index(drop=True); DHe, PHe = DHe[~overlap], PHe[~overlap]

        # --- groups for the CI ---
        if cfg["patient_regex"]:
            pid = edf.filename.str.extract(cfg["patient_regex"])[0]
            assert pid.notna().all(), "patient_regex missed some filenames"
            edf["group_id"] = pd.factorize(pid)[0]; gkind = "patient ID"
        else:
            hd, hp = hamming(DHe, DHe), hamming(PHe, PHe)
            adj = coo_matrix((hd <= DHASH_T) & (hp <= PHASH_T))
            _, edf["group_id"] = connected_components(adj, directed=False); gkind = "DAP hash group"
        print(f"  groups: {edf.group_id.nunique()} ({gkind})")

        # --- predict ---
        y = edf.label.values; g = edf.group_id.values; preds = []
        for name, spec in models.items():
            print(f"  {name}")
            lg = spec["fn"](edf.path.tolist()); pr = softmax(lg, spec["T"]); pd_ = pr.argmax(1)
            b = binary(y, pr, pd_); ci = binary_ci(y, pr, pd_, g)
            rows.append({"cohort": cname, "model": name, "n_images": len(edf), "n_groups": int(len(np.unique(g))),
                         "group_kind": gkind, **b,
                         **{f"{k}_ci": f"[{lo:.3f}, {hi:.3f}]" for k, (lo, hi) in ci.items()},
                         **{f"pred_{c}": float((pd_ == i).mean()) for i, c in enumerate(CLASSES)}})
            print(f"    Normal recall {b['normal_recall']:.3f} | Stone sens {b['stone_sensitivity']:.3f} | "
                  f"bal-acc {b['balanced_accuracy']:.3f} | AUC {b['auc_p_stone']:.3f}")
            preds.append(edf[["path", "filename", "class_name", "group_id"]].assign(
                model=name, pred=np.array(CLASSES)[pd_], conf=pr.max(1),
                **{f"p_{c}": pr[:, i] for i, c in enumerate(CLASSES)}))
        pd.concat(preds).to_csv(OUT / f"{cname}_predictions.csv", index=False)
        summary[cname] = {"status": "evaluated", "desc": cfg["desc"], "n_images_evaluated": int(len(edf)),
                          "class_counts": {k: int(v) for k, v in edf.class_name.value_counts().items()},
                          "n_groups": int(edf.group_id.nunique()), "group_kind": gkind,
                          "excluded_overlap": int(overlap.sum()), "smoke_test_limit": args.limit}

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "external_validation_summary.csv", index=False)
    (OUT / "external_validation_run.json").write_text(json.dumps(summary, indent=2))
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print("\n" + res[["cohort", "model", "n_images", "n_groups", "normal_recall", "stone_sensitivity",
                       "specificity", "balanced_accuracy", "auc_p_stone"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
